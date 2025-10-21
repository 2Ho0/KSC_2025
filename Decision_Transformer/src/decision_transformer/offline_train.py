import pytest
import torch as t
import torch.nn as nn
from einops import rearrange
from dataclasses import asdict
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from collections import deque
import sys
import os

import wandb
import ruamel.yaml as yaml
from Decision_Transformer.src.config import EnvironmentConfig, OfflineTrainConfig
from Decision_Transformer.src.models.trajectory_transformer import (
    TrajectoryTransformer,
)

from .offline_dataset import TrajectoryDataset
from .eval import evaluate_dt_agent
from .utils import configure_optimizers, get_scheduler
from torch.utils.data import ConcatDataset
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
from torch.utils.data import Sampler
from typing import List, Iterator

def offline_train(
    model: TrajectoryTransformer,
    trajectory_data_set: TrajectoryDataset,
    env,
    offline_config: OfflineTrainConfig,
    device="cpu",
    
):
    # 전체 학습 데이터의 task_label 수집 필요
  
    # 각 태스크별 데이터셋 수 출력
    print("\n===== 태스크별 데이터셋 크기 =====")
    for task_id, dataset in trajectory_data_set.items():
        print(f"Task {task_id}: {len(dataset)} 샘플")

    train_dataloader, test_dataloader, train_subsets_info = get_dataloaders(
        trajectory_data_set, offline_config
    )

    task_labels_list = []
    # train_subsets_info는 {task_id: length} 형태의 dict
    for task_id, length in sorted(train_subsets_info.items()):
        task_labels_list.extend([task_id] * length)
    unique_classes = np.unique(task_labels_list) 

    # 찾은 unique_classes를 classes 인자로 전달합니다.
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=task_labels_list)

    print("Detected Unique Classes:", unique_classes) # 클래스가 제대로 인식되었는지 확인
    print("Class Weights:", class_weights)

    weight_tensor = t.tensor(class_weights, dtype=t.float32).to(device)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    mode = offline_config.mode

    
    
    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]

    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )
    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(train_dataloader):

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n  # dummy action for padding

            optimizer.zero_grad()

           
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            state_preds, action_preds, reward_preds= model(
                states=s, # (64,21,6,6,3)
                actions=action, # (64,20,1)
                rtgs=rtg,  # (64,21,1)
                timesteps=ti.unsqueeze(-1), # (64,21)
                mlp_learn=False,
            )

            if mode == 'state':
                state_preds = rearrange(state_preds, "b t s -> (b t) s") # 128, 4, 7, 7, 20
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)
                loss = nn.MSELoss()(state_preds, s_exp)

            elif mode == 'action':
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)
                mask = a_exp != env.action_space.n
                loss = loss_fn(action_preds[mask], a_exp[mask])

            elif mode == 'rtg':
                r = r[:, 1:] # 128, 100, 1
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s") # 12800, 1
                r_exp = rearrange(r.squeeze(-1), "b t -> (b t)").to(t.float32)
                loss = nn.MSELoss()(reward_preds.squeeze(-1), r_exp)
           
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

    # MLP training start
    # Step 2: Freeze all except MLP layers (penultimate_layer, output_layer)
    print("\n🔒 Freezing all layers except MLP (penultimate_layer, output_layer)")
    for name, param in model.named_parameters():
        if "penultimate_layer" in name or "output_layer" in name:
            param.requires_grad = True
            print(f"✅ {name} will be updated.")
        else:
            param.requires_grad = False
            print(f"❌ {name} is frozen.")

    # 새로운 옵티마이저와 스케줄러 설정
    optimizer = configure_optimizers(model, offline_config)
    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )

    # MLP만을 위한 추가 학습 루프
    pbar_mlp = tqdm(range(offline_config.mlp_train_epochs), desc="MLP Fine-Tuning")
    for epoch in pbar_mlp:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(train_dataloader):
            model.train()
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None

            state_preds, action_preds, reward_preds, task_preds, _ = model(
                states=s,
                actions=action,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                mlp_learn=True,
            )

            # task classification loss만 사용
            task_labels = task_id.to(task_preds.device)
            preds_for_loss = task_preds[:, -1, :]
            task_loss = model.label_smoothing_loss(preds_for_loss, task_labels, class_weights=weight_tensor)

            task_pred = t.argmax(task_preds, dim=-1)
            final_timestep_preds = task_preds[:, -1, :] # -> shape: [64, 3]

            # 2. 선택된 2D 텐서에 대해 argmax를 취해 최종 예측 레이블을 구합니다.
            task_pred = t.argmax(final_timestep_preds, dim=-1)
            n_correct = (task_pred == task_labels).sum().item()

            n_total = task_labels.shape[0]
            task_accuracy = n_correct / n_total

            # task accuracy
            task_correct = {}
            task_total = {}
            for pred, true in zip(task_pred.cpu(), task_labels.cpu()):
                true = int(true)
                if true not in task_correct:
                    task_correct[true] = 0
                    task_total[true] = 0
                task_correct[true] += int(pred == true)
                task_total[true] += 1

            if offline_config.track:
                wandb.log({
                    "train/MLP_loss": task_loss.item(),
                    "train/MLP_accuracy": task_accuracy,
                })
                for tid in sorted(task_correct.keys()):
                    acc = task_correct[tid] / task_total[tid]
                    wandb.log({f"train/MLP_task{tid}_accuracy": acc})

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar_mlp.set_description(f"MLP Fine-Tuning: task_loss = {task_loss.item():.4f}")

    result = test(
        model=model,
        dataloader=test_dataloader,
        env=env,
        epochs=offline_config.test_epochs,
        track=offline_config.track,
        # batch_number=batch_number,
        mode = mode,
        class_weights=weight_tensor,
        device=device
    )
    return result


@pytest.mark.skip(reason="This is not a test")
def test(
    model: TrajectoryTransformer,
    dataloader: DataLoader,
    env,
    epochs=10,
    track=True,
    batch_number=0,
    mode="rtg",
    class_weights=None,
    device="cpu",
):

    model.eval()

    loss_fn = nn.CrossEntropyLoss()
        
    main_loss = 0
    main_total = 0
    main_correct = 0

    task_loss = 0
    n_task_correct = 0
    n_task_total = 0

    all_task_preds = []
    all_task_labels = []
    all_embeddings = []
    task_correct = {}
    task_total = {}

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, (s, a, r, d, rtg, ti, m, _, task_id) in enumerate(dataloader):
            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)

            a[a == -10] = env.action_space.n
            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            with t.no_grad():
                state_preds, action_preds, reward_preds, task_preds, penultimate_out = model(
                    states=s,
                    actions=action,
                    rtgs=rtg,
                    timesteps=ti.unsqueeze(-1),
                    mlp_learn=True,
                )

            if mode == "state":
                state_preds = rearrange(state_preds, "b t s -> (b t) s")
                s_exp = rearrange(s[:, 1:], "b t h w c -> (b t) (h w c)").to(t.float32)

                main_loss += nn.MSELoss()(state_preds, s_exp).item()
                main_total += s_exp.shape[0]

            elif mode == "action":
                action_preds = rearrange(action_preds, "b t a -> (b t) a")
                a_exp = rearrange(a[:, 1:], "b t -> (b t)").to(t.int64)

                mask = a_exp != env.action_space.n
                action_preds = action_preds[mask]
                a_exp = a_exp[mask]
                a_hat = t.argmax(action_preds, dim=-1)

                main_loss += loss_fn(action_preds, a_exp).item()
                main_total += a_exp.shape[0]
                main_correct += (a_hat == a_exp).sum().item()

            elif mode == "rtg":
                reward_preds = rearrange(reward_preds, "b t s -> (b t) s")
                r_exp = rearrange(r[:, 1:].squeeze(-1), "b t -> (b t)").to(t.float32)

                main_loss += nn.MSELoss()(reward_preds.squeeze(-1), r_exp).item()
                main_total += r_exp.shape[0]

            embeddings = penultimate_out  # shape: (B, D)
            all_embeddings.append(embeddings.cpu())

            task_ids_expanded = match_task_ids(task_id, embeddings)
            all_task_labels.extend(task_ids_expanded.cpu().tolist())

            


            # ✅ Task classification evaluation
            if task_id is not None:
                
                task_labels = task_id.to(task_preds.device)
                preds_for_loss = task_preds[:, -1, :]
                task_loss += model.label_smoothing_loss(preds_for_loss, task_labels, class_weights=class_weights).item()
                
                task_pred = t.argmax(task_preds, dim=-1)
                final_timestep_preds = task_preds[:, -1, :]
                task_pred = t.argmax(final_timestep_preds, dim=-1)
                n_task_correct += (task_pred == task_labels).sum().item()
                n_task_total += task_labels.shape[0]

                # 분포 기록
                all_task_preds.extend(task_pred.cpu().tolist())

                # 개별 task 정확도 기록
                for pred, true in zip(task_pred.cpu(), task_labels.cpu()):
                    true = int(true)
                    if true not in task_correct:
                        task_correct[true] = 0
                        task_total[true] = 0
                    task_correct[true] += int(pred == true)
                    task_total[true] += 1
                    
    mean_main_loss = main_loss / (epochs * test_batches_per_epoch)
    main_accuracy = main_correct / main_total if mode == "action" else None

    mean_task_loss = task_loss / (epochs * test_batches_per_epoch)
    mean_task_accuracy = n_task_correct / n_task_total

    # ✅ Print logs
    print(f"\n==== [Test Summary] ====")
    print(f"{mode} loss: {mean_main_loss:.4f}")
    if mode == "action":
        print(f"{mode} accuracy: {main_accuracy:.4f}")
    
    print(f"Task loss: {mean_task_loss:.4f}")
    print(f"Task accuracy: {mean_task_accuracy:.4f}")
    for tid in sorted(task_correct.keys()):
        acc = task_correct[tid] / task_total[tid]
        print(f"- Task {tid}: {acc:.4f} ({task_correct[tid]}/{task_total[tid]})")

    # wandb 로그
    wandb.log({f"test/{mode}_loss": mean_main_loss}, step=batch_number)
    if mode == "action":
        wandb.log({f"test/{mode}_accuracy": main_accuracy}, step=batch_number)

    if track:
        wandb.log({
            "test/task_loss": mean_task_loss,
            "test/task_accuracy": mean_task_accuracy,
            "test/task_pred_distribution": wandb.Histogram(all_task_preds),
            "test/task_true_distribution": wandb.Histogram(all_task_labels),
        })

        for task_id in task_total:
            acc = task_correct[task_id] / task_total[task_id]
            wandb.log({f"test/task{task_id}_accuracy": acc}, step=batch_number)



    all_embeddings_tensor = t.cat(all_embeddings, dim=0)  # ¡æ (N, D)
    all_task_labels_tensor = t.tensor(all_task_labels)
    print("all_task_labels: ", len(all_task_labels))
    print("all_task_labels_tensor: ", all_task_labels_tensor.shape)

    return {
        "model": model,
        "embeddings": all_embeddings_tensor,  # (NEW)
        "task_ids": all_task_labels_tensor,
    }



class TaskBatchSampler(Sampler[List[int]]):
    """
    배치 내에서는 동일한 태스크의 데이터만 포함하고,
    배치들의 순서는 무작위로 섞는 샘플러입니다.

    Args:
        - datasets_lengths (List[int]): 각 태스크별 데이터셋의 길이를 담은 리스트
        - batch_size (int): 배치 크기
        - drop_last (bool): 마지막 배치가 batch_size보다 작을 경우 버릴지 여부
    """
    def __init__(self, datasets_lengths: List[int], batch_size: int, drop_last: bool):
        self.datasets_lengths = datasets_lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

        # ConcatDataset에서의 각 데이터셋의 시작 인덱스를 계산합니다.
        self.cumulative_sizes = [0] + list(np.cumsum(datasets_lengths))
        
        # 모든 태스크에 대해 가능한 모든 배치를 미리 생성합니다.
        self.batches = []
        for i in range(len(datasets_lengths)):
            start_idx = self.cumulative_sizes[i]
            end_idx = self.cumulative_sizes[i+1]
            indices = list(range(start_idx, end_idx))
            
            # 각 태스크 내에서 인덱스를 섞어 다양성을 확보합니다.
            random.shuffle(indices)

            # 배치 생성
            for j in range(0, len(indices), self.batch_size):
                batch_indices = indices[j : j + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                self.batches.append(batch_indices)

    def __iter__(self) -> Iterator[List[int]]:
        # 생성된 모든 배치들의 순서를 무작위로 섞습니다.
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        # 전체 배치의 개수를 반환합니다.
        return len(self.batches)

def get_dataloaders(trajectory_data_set, offline_config):
    """
    trajectory_data_set: Dict[int, Dataset]
    """
    if isinstance(trajectory_data_set, ConcatDataset):
        raise ValueError("입력은 ConcatDataset이 아닌 Dict[int, Dataset] 형태여야 합니다.")

    # 1. 각 태스크별로 train/test 데이터셋으로 분리
    train_subsets = []
    test_subsets = []
    # 일관된 순서를 위해 task_id를 정렬합니다.
    sorted_task_ids = sorted(trajectory_data_set.keys()) 
    train_subsets_info = {}

    for task_id in sorted_task_ids:
        dataset = trajectory_data_set[task_id]
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        
        # 시드를 고정하여 항상 동일한 분할이 이루어지도록 합니다.
        train_subset, test_subset = random_split(
            dataset,
            [train_size, test_size],
            generator=t.Generator().manual_seed(42)
        )
        train_subsets.append(train_subset)
        test_subsets.append(test_subset)
        train_subsets_info[task_id] = len(train_subset)

    # 2. 분리된 Subset들을 하나의 ConcatDataset으로 합침
    train_concat_dataset = ConcatDataset(train_subsets)
    test_concat_dataset = ConcatDataset(test_subsets)
    
    print(f"Total train samples: {len(train_concat_dataset)}")
    print(f"Total test samples: {len(test_concat_dataset)}")

    # 3. TaskBatchSampler 생성
    train_sampler = TaskBatchSampler(
        datasets_lengths=[len(ds) for ds in train_subsets],
        batch_size=offline_config.batch_size,
        drop_last=True
    )
    test_sampler = TaskBatchSampler(
        datasets_lengths=[len(ds) for ds in test_subsets],
        batch_size=offline_config.batch_size,
        drop_last=False
    )

    # 4. DataLoader에 batch_sampler를 적용
    # batch_sampler를 사용하면 batch_size, shuffle, sampler, drop_last를 지정하면 안 됩니다.
    train_dataloader = DataLoader(
        train_concat_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_concat_dataset,
        batch_sampler=test_sampler,
        num_workers=4,
    )

    return train_dataloader, test_dataloader, train_subsets_info

def match_task_ids(task_id, embedding_tensor):
    N = embedding_tensor.shape[0]
    B = task_id.shape[0]
    repeats = N // B
    expanded = task_id.repeat_interleave(repeats)

    if expanded.shape[0] > N:
        expanded = expanded[:N]
    elif expanded.shape[0] < N:
        print(f"?? Padding task_ids with last label. ({expanded.shape[0]} ¡æ {N})")
        pad = t.full((N - expanded.shape[0],), expanded[-1], dtype=expanded.dtype)
        expanded = t.cat([expanded, pad])
    return expanded