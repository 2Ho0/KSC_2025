import collections
import functools
from functools import partial as bind

import elements
import embodied
import numpy as np
import torch as t

from Decision_Transformer.src.decision_transformer.train import dt_inference
from Decision_Transformer.src.models.trajectory_transformer import (DecisionTransformer)
from Decision_Transformer.src.config import( EnvironmentConfig, TransformerModelConfig)
from Decision_Transformer.src.decision_transformer.offline_dataset import TrajectoryDataset

import jax
from gymnasium.spaces import Box, Discrete

  
def create_dreamer_batch_from_dt(dt_batch_np, reference_batch_jax):
    """DT 배치를 기반으로 Dreamer 학습에 필요한 배치 구조를 생성합니다."""
    
    # 새로운 배치를 NumPy 배열로 구성할 딕셔너리
    new_batch_np = {}
    
    # 1. 직접 매핑 가능한 키들
    new_batch_np['image'] = dt_batch_np['states']
    new_batch_np['action'] = dt_batch_np['actions']
    # new_batch_np['reward'] = dt_batch_np['rewards']
    new_batch_np['is_last'] = dt_batch_np['is_last']
    new_batch_np['is_terminal'] = dt_batch_np['is_last'] # is_last와 동일하게 가정
    new_batch_np['episode_step'] = dt_batch_np['timesteps']
    # new_batch_np['task_id'] = dt_batch_np['task_ids']
    new_batch_np['rtg'] = dt_batch_np['returns_to_go']  # Dreamer에서 RTG를 rtg로 사용

    is_last_shifted = np.roll(new_batch_np['is_last'], 1, axis=1)
    is_last_shifted[:, 0] = True
    new_batch_np['is_first'] = is_last_shifted
    
    # 3. Dreamer에 필수적이지만 DT 배치에는 없는 키들 (Zero-padding)
    # 모델의 내부 상태(dyn/*)는 정보가 없으므로 0으로 초기화하여, 
    # "여기서부터 새로운 상상을 시작하라"는 신호를 줍니다.
    for key, value in reference_batch_jax.items():
        if key not in new_batch_np:
            # 원본 배치의 shape과 dtype을 참조하여 0으로 채운 배열 생성
            new_batch_np[key] = np.zeros(value.shape, dtype=value.dtype)

    # 4. NumPy 딕셔너리를 JAX 딕셔너리로 변환
    return jax.tree_util.tree_map(jax.device_put, new_batch_np)
    # return new_batch_np

def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  task_manager = None
  if isinstance(make_env, functools.partial) and 'task_manager' in make_env.keywords:
      task_manager = make_env.keywords['task_manager']
  agent = make_agent()
  logger = make_logger()

  # Dreamer 학습을 위한 통합 리플레이 버퍼 (모든 태스크 데이터가 여기에 모임)
  replay = make_replay() 
  dt_replay = make_replay(mode='decision_transformer')

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  episode_buffers = collections.defaultdict(list)

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  # Decision Transformer 모델 및 관련 설정 초기화 (학습 루프 시작 전 1회 실행)
  dt_env_config = EnvironmentConfig()
  dt_model_config = TransformerModelConfig()

  # DT용 환경은 DT 모델 내부의 observation/action space 정보를 설정하기 위해 한 번만 생성합니다.
  dt_env = make_env(0)  # 'episode_step' 추가
  act_space = dt_env.act_space
  
  # embodied의 딕셔너리 형태에서 실제 Space 객체를 추출합니다.
  main_action_space = act_space['action']
  num_actions = int(main_action_space.high) # .high가 행동의 개수를 나타냅니다.
  dt_env_config.action_space = Discrete(num_actions)
  mock_obs_space = Box(low=0, high=255, shape=(6, 6, 3), dtype=np.uint8)
  dt_env_config.observation_space = mock_obs_space
  
  dt_model = DecisionTransformer(
      environment_config=dt_env_config,
      transformer_config=dt_model_config,
  )
  model_path = '/home/hail/Desktop/Alarm/models/8377.pt'
  checkpoint = t.load(model_path)
  dt_model.load_state_dict(checkpoint['model_state_dict'])

  def calculate_rtg(rewards, gamma=0.99):
    """주어진 보상 시퀀스로 Returns-To-Go를 계산합니다."""
    rtgs = []
    discounted_reward = 0
    for r in reversed(rewards):
        discounted_reward = r + gamma * discounted_reward
        rtgs.insert(0, discounted_reward)
    return np.array(rtgs, dtype=np.float32)

  def collect_and_process_episode(tran, worker):
      # 1. 현재 스텝의 데이터를 해당 워커의 임시 버퍼에 추가
      episode_buffers[worker].append(tran)

      buffer_len = len(episode_buffers[worker])
      if buffer_len % 700 == 0: # 700 스텝마다 출력
          print(f"DEBUG: Worker {worker}, episode_buffer length: {buffer_len}")
      
      # 2. 에피소드가 끝나지 않았으면 아무것도 하지 않고 종료
      if not tran['is_last']:
          return

      # 3. 에피소드가 끝났으면, 해당 워커의 버퍼에서 모든 데이터를 가져옴
      episode_data = episode_buffers.pop(worker)
      
      # 4. 보상(reward)만 추출하여 RTG 계산
      rewards = [step['reward'] for step in episode_data]
      rtgs = calculate_rtg(rewards) # RTG 계산 함수 호출

      # TaskManager로부터 현재 활성 태스크의 인덱스를 가져옴
      current_task_index = task_manager.current_task_idx % len(task_manager.tasks)
      
      # 5. 에피소드의 모든 스텝에 대해 RTG와 추가 정보를 붙여 리플레이 버퍼에 추가
      for i, step_data in enumerate(episode_data):
          # tran은 수정하면 안 되므로 복사본을 만듦
          labeled_tran = dict(step_data)
          
          # 계산된 RTG와 태스크 ID, 글로벌 스텝을 추가
          labeled_tran['rtg'] = np.array([rtgs[i]], dtype=np.float32)
          labeled_tran['task_id'] = np.array(current_task_index, dtype=np.int32)
          labeled_tran['global_step'] = np.array(step.value, dtype=np.int64)
          
          # 라벨이 붙은 데이터를 각 리플레이 버퍼에 추가
          replay.add(labeled_tran, worker)
          dt_replay.add(labeled_tran, worker)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          # episode.add(f'policy_{key}', value, agg='stack')
          continue
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

      if task_manager:
        # 1. TaskManager에 에피소드 종료를 알립니다.
        task_manager.on_episode_end(
            episode_reward=result.get('score', 0),
            episode_length=result.get('length', 0)
        )
        # 2. 태스크를 전환해야 하는지 확인합니다.
        if task_manager.should_switch_task():
          old_task = task_manager.get_current_task_name()
          task_manager.switch_task()
          new_task = task_manager.get_current_task_name()
          print(f"🔄 Task 전환: {old_task} -> {new_task}. 환경을 리셋합니다.")

          # # 3. 모든 환경을 리셋하여 새로운 태스크를 적용합니다.
          # driver.reset(agent.init_policy)
          
          # # 4. 새로운 태스크에 대한 통계를 새로 기록하기 위해 초기화합니다.
          # episodes.clear()
          # epstats.reset()
          # carry_train[0] = agent.init_train(args.batch_size)

  fns = [bind(make_env, i, task_manager=task_manager) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(collect_and_process_episode) 
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      # Decision Transformer로 Task Shift 감지
      with elements.timer.section('dt_task_shift'):
        dt_batch = dt_replay.sample(args.batch_size)
        # batch_dt = jax.device_get(dt_batch)
        batch_dt = dt_batch
        
        B, T = batch_dt['is_first'].shape
        
        dt_batch_data = {
            'states': batch_dt['image'], # (64,21,6,6,3)
            'actions': batch_dt['action'], # (64,21)
            # 'rewards': batch_dt['reward'], # (64,21)
            'returns_to_go': batch_dt['rtg'], # (64,21)
            # 'attention_mask': ~batch_dt['is_last'],
            # 'attention_mask': np.tril(np.ones((T, T))).astype(bool), # (21,21)
            'timesteps': batch_dt['episode_step'], # (64,21)
            # 'task_ids': batch_dt['task_id'], # (64,21)
            'is_last': batch_dt['is_last'] # (64,21)
        }
        print("returns_to_go:", dt_batch_data['returns_to_go'].max())
        
        task_shift = dt_inference(
              model=dt_model,
              trajectory_data_set=dt_batch_data,
              num_actions=num_actions
          )
        
        current_task_in_batch = int(np.bincount(batch_dt['task_id'].flatten()).argmax())
        
        if task_shift:
            # 위에서 정의한 헬퍼 함수를 사용해 DT 배치를 Dreamer 형식으로 변환
            train_batch = create_dreamer_batch_from_dt(dt_batch_data, batch)
            # train_batch = batch_dt
            train_batch['seed'] = batch['seed']
            train_batch['consec'] = batch['consec']
            carry_train[0], outs, mets = agent.train_real(carry_train[0], train_batch)
        else:
            # Task Shift가 없으면 기존 Dreamer 배치를 그대로 사용
            train_batch = batch
            carry_train[0], outs, mets = agent.train(carry_train[0], train_batch)
        
        train_fps.step(batch_steps)
        if 'replay' in outs:
          replay.update(outs['replay'])
        train_agg.add(mets, prefix='train')

  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=100)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        batch_report = next(stream_report)
        current_task_index = task_manager.current_task_idx
        B, T_rep = batch_report['is_first'].shape
        # task_shift_result: 리포트에선 False로 채워도 OK
        # batch_report['task_shift_result'] = np.zeros((B, T_rep, T_rep), dtype=bool)
        # task_id: 현재 태스크 ID로 채움
        batch_report['task_id'] = np.full((B, T_rep), int(current_task_index), dtype=np.int32)

        carry_report, mets = agent.report(carry_report, batch_report)
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()