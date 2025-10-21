import os
from typing import Optional, Union
import numpy as np

from gymnasium.vector import SyncVectorEnv
from tqdm.autonotebook import tqdm

import wandb
from config import (
    EnvironmentConfig,
    LSTMModelConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)

from .agent import PPOAgent, get_agent
from .memory import Memory
from .utils import store_model_checkpoint

from ppo.agent import RandomAgent

def train_ppo(
    run_config: RunConfig,
    online_config: OnlineTrainConfig,
    environment_config: EnvironmentConfig,
    model_config: Optional[Union[TransformerModelConfig, LSTMModelConfig]],
    envs: SyncVectorEnv,
    trajectory_writer=None,
) -> PPOAgent:
    """
    PPO 에이전트를 주어진 환경에서 학습하는 함수입니다.
    """

    # Rollout 데이터를 저장할 메모리 버퍼 초기화
    memory = Memory(envs, online_config, run_config.device)

    # 에이전트 초기화 (Transformer/LSTM 기반 PPOAgent)
    agent = get_agent(
    model_config,
    envs=envs,
    environment_config=environment_config,
    online_config=online_config
)
    # 전체 업데이트 횟수 계산
    num_updates = online_config.total_timesteps // online_config.batch_size

    # 옵티마이저 및 러닝레이트 스케줄러 설정
    optimizer, scheduler = agent.make_optimizer(
        num_updates=num_updates,
        initial_lr=online_config.learning_rate,
        end_lr=online_config.learning_rate if not online_config.decay_lr else 0.0,
    )

    checkpoint_num = 1
    if run_config.track:
        # wandb artifact (모델 체크포인트) 생성
        checkpoint_artifact = wandb.Artifact(
            f"{run_config.exp_name}_checkpoints", type="model"
        )

        # 몇 번의 update마다 checkpoint 저장할지 설정
        checkpoint_interval = num_updates // online_config.num_checkpoints + 1

        # 초기 체크포인트 저장
        checkpoint_num = store_model_checkpoint(
            agent, online_config, run_config, checkpoint_num, checkpoint_artifact
        )

    # 학습 진행 상황 출력용 progress bar
    progress_bar = tqdm(range(num_updates), position=0, leave=True)
    for n in progress_bar:
        # 환경에서 rollout 데이터를 수집하고 메모리에 저장
        agent.rollout(memory, online_config.num_steps, envs, trajectory_writer)

        # 메모리로부터 학습 수행 (policy, value 업데이트)
        agent.learn(memory, online_config, optimizer, scheduler, run_config.track)

        if run_config.track:
            # wandb 로깅 (리턴, 길이 등)
            memory.log()

            # checkpoint 저장 주기에 도달하면 저장
            if (n + 1) % checkpoint_interval == 0:
                checkpoint_num = store_model_checkpoint(
                    agent, online_config, run_config, checkpoint_num, checkpoint_artifact
                )

        # 현재 리턴 등 통계 정보 출력
        output = memory.get_printable_output()
        progress_bar.set_description(output)

        # 메모리 초기화
        memory.reset()

    if run_config.track:
        # 마지막 체크포인트 저장 및 wandb 업로드
        checkpoint_num = store_model_checkpoint(
            agent, online_config, run_config, checkpoint_num, checkpoint_artifact
        )
        wandb.log_artifact(checkpoint_artifact)

    if trajectory_writer is not None:
        # trajectory 종료 마킹 및 저장
        trajectory_writer.tag_terminated_trajectories()
        trajectory_writer.write(upload_to_wandb=run_config.track)

    # 환경 종료
    envs.close()

    # 학습된 에이전트 반환
    return agent

# def count_successful_episodes(trajectory_writer):
#     count = 0
#     for reward_seq, done_seq, truncated_seq in zip(
#         trajectory_writer.rewards, trajectory_writer.dones, trajectory_writer.truncated
#     ):
#         # 성공 조건: reward 합 > 0이고, 종료되었거나 잘렸을 경우
#         # if np.sum(reward_seq) > 0.0 and (done_seq[-1] or truncated_seq[-1]):
#         if np.sum(reward_seq) > 0.0:
#             count += 1
#     return count

def train_random(
    run_config,
    online_config,
    environment_config,
    envs,
    model_config="random",
    trajectory_writer=None,
) -> PPOAgent:
    agent = get_agent(
        model_config=model_config,
        envs=envs,
        environment_config=environment_config,
        online_config=online_config
    )
    memory = Memory(envs, online_config, run_config.device)
    num_updates = online_config.total_timesteps // online_config.num_steps
    
    checkpoint_num = 1
    # if run_config.track:
    #     checkpoint_artifact = wandb.Artifact(f"{run_config.exp_name}_checkpoints", type="model")
    #     checkpoint_interval = max(1, num_updates // online_config.num_checkpoints)

    progress_bar = tqdm(range(num_updates), position=0, leave=True)
    for n in progress_bar:

        agent.rollout(memory, online_config.num_steps, envs, trajectory_writer)
        agent.learn(memory, online_config, optimizer=None, scheduler=None, track=run_config.track)

        # if run_config.track:
        #     memory.log()
        #     if (n+1) % checkpoint_interval == 0:
        #         checkpoint_num = store_model_checkpoint(
        #             agent, online_config, run_config, checkpoint_num, checkpoint_artifact
        #         )
        
        output = memory.get_printable_output()
        progress_bar.set_description(output)
        memory.reset()

    if trajectory_writer is not None:
        try:
            env_id = envs.envs[0].spec.id
            if "DoorKey" in env_id:
                task_id = 0
            elif "LavaCrossing" in env_id:
                task_id = 1
            elif "SimpleCrossing" in env_id:
                task_id = 2
            else:
                task_id = -1
        except:
            task_id = getattr(envs.envs[0], "task_id", 0)

        trajectory_writer.add_metadata({"task_id": task_id})
        trajectory_writer.tag_terminated_trajectories()

        trajectory_writer.write(upload_to_wandb=run_config.track)
        trajectory_writer.reset()

    # if run_config.track:
    #     checkpoint_num = store_model_checkpoint(
    #         agent,
    #         online_config,
    #         run_config,
    #         checkpoint_num,
    #         checkpoint_artifact,
    #     )
    #     wandb.log_artifact(checkpoint_artifact)  # Upload checkpoints to wandb

    envs.close()
    return agent

def check_and_upload_new_video(video_path, videos, step=None):
    """
    Checks if new videos have been generated in the video path directory since the last check, and if so,
    uploads them to the current WandB run.

    Args:
    - video_path: The path to the directory where the videos are being saved.
    - videos: A list of the names of the videos that have already been uploaded to WandB.
    - step: The current step in the training loop, used to associate the video with the correct timestep.

    Returns:
    - A list of the names of all the videos currently present in the video path directory.
    """

    current_videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    new_videos = [i for i in current_videos if i not in videos]
    if new_videos:
        for new_video in new_videos:
            path_to_video = os.path.join(video_path, new_video)
            # wandb.log(
            #     {
            #         "video": wandb.Video(
            #             path_to_video,
            #             fps=4,
            #             caption=new_video,
            #             format="mp4",
            #         )
            #     },
            #     step=step,
            # )
    return current_videos

def prepare_video_dir(video_path):
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    for video in videos:
        os.remove(os.path.join(video_path, video))
    videos = []