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

# 이 함수는 dt_train.py 또는 utils.py 같은 파일에 추가하면 좋습니다.

def dt_inference(model, trajectory_data_set, num_actions, device="cpu"):
    """학습된 DT 모델로 태스크 전환 여부만 추론하는 함수"""
    model.eval()
    model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # s, a, _, _, rtg, ti, _, _, _ = next(iter(trajectory_data_set))
    # s, a, rtg, ti = s.to(device), a.to(device), rtg.to(device), ti.to(device)
    s   = t.from_numpy(trajectory_data_set['states']).to(device)         # (B,T,...)
    a   = t.from_numpy(trajectory_data_set['actions']).to(device)        # (B,T)
    rtg = t.from_numpy(trajectory_data_set['returns_to_go']).to(device)  # (B,T)
    ti = t.from_numpy(np.array(trajectory_data_set['timesteps'])).to(device=device, dtype=t.long)      # (B,T)
    
    a[a == -10] = num_actions

    # 옵티마이저, 손실 계산, 역전파가 전혀 필요 없음
    with t.no_grad():
        # mlp_learn=True로 task_preds를 바로 얻음
        _, _, _, task_preds, _ = model(             # task_preds(64,21,3)
            states=s,                            # s.shape (64,21, 6, 6, 3)
            actions=a,                          # a.shape(64,21)
            rtgs=rtg,                       # rtg.shape(64,21)   
            timesteps=ti.unsqueeze(-1),     # ti.shape(64,21)
            mlp_learn=True,
        )

        # 태스크 전환 감지 로직
        task_probs = t.softmax(task_preds, dim=-1)
        cur_task_seq = task_probs.argmax(dim=-1)  
        
        # 배치별 현재 태스크 (B,)
        batch_current_tasks = cur_task_seq.mode(dim=1).values
        all_same = (batch_current_tasks == batch_current_tasks[0]).all()
        task_shift_detected = not bool(all_same)
        # # 배치별 전환 여부 (B,)
        # batch_shift_mask = (batch_current_tasks != int(pre_task))
        # # 전체 중 하나라도 전환됐는지
        # task_shift_detected = bool(batch_shift_mask.any().item())

    # 현재 태스크와 전환 여부를 함께 반환하여 다음 스텝에서 pre_task를 업데이트
    return task_shift_detected
