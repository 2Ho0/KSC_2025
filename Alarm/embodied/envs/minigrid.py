from typing import cast
import gymnasium
from minigrid.wrappers import FullyObsWrapper
from .from_gym import FromGymnasium
from gymnasium import spaces
import numpy as np
from PIL import Image

from typing import cast
import gymnasium
# from gymnasium.core import ObservationWrapper
from minigrid.wrappers import ObservationWrapper
from gymnasium import spaces
import elements
import embodied

class EpisodeStepWrapper(embodied.Env):
    """
    에피소드 내의 현재 스텝 수를 세어서 'episode_step' 키로 관측값에 추가하는 래퍼.
    에피소드가 시작되면(reset) 카운터는 0으로 초기화됩니다.
    """
    def __init__(self, env):
        self._env = env
        self._step = 0

    @property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        # [수정] 데이터 타입을 float32로 명시하여 one-hot 인코딩 시도를 방지합니다.
        spaces['episode_step'] = elements.Space(np.float32)
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        self._step += 1
        if obs['is_first']:
            self._step = 0
        # [수정] 반환되는 값도 float32로 변환합니다.
        obs['episode_step'] = np.array(self._step, dtype=np.float32)
        # [수정] obs를 반드시 반환해야 데이터 수집이 계속됩니다.
        return obs
    
class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""
    def __init__(self, env):
        super().__init__(env)
        old = cast(gymnasium.spaces.Dict, self.observation_space)
        # 새 Dict로 재할당 (in-place pop 금지)
        new_spaces = {k: v for k, v in old.spaces.items() if k != 'mission'}
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation: dict):
        # obs에서도 실제 키 제거
        if 'mission' in observation:
            observation = dict(observation)
            observation.pop('mission', None)
        return observation

# class HideMission(ObservationWrapper):
#     """Remove the 'mission' string from the observation."""
#     def __init__(self, env):
#         super().__init__(env)
#         obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
#         obs_space.spaces.pop('mission', None)

#     def observation(self, observation: dict):
#         observation.pop('mission', None)
#         return observation

class Minigrid(FromGymnasium):
    def __init__(self, task: str, fully_observable: bool, hide_mission: bool):
        env = gymnasium.make(f"{task}-v0", render_mode="rgb_array")
        if fully_observable:
            env = FullyObsWrapper(env)
        if hide_mission:
            env = HideMission(env)
        env = ViewSizeWrapper(env, agent_view_size=6)
        super().__init__(env=env)


class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        # assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size
        current_dim = self.observation_space["image"].shape[2:]
        # Compute observation space with specified view size
        new_image_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, *current_dim),
            dtype="uint8",
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}

# class ResizeIntObservation(ObservationWrapper):
#     def __init__(self, env, size=(64, 64)):
#         super().__init__(env)
#         self.size = size
#         self.observation_space.spaces['image'] = spaces.Box(
#             low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8
#         )

#     def observation(self, obs):
#         image = obs['image']  # (H, W, 3)
#         resized_channels = []
#         for c in range(image.shape[-1]):
#             channel = Image.fromarray(image[:, :, c]).resize(self.size, resample=Image.NEAREST)
#             resized_channels.append(np.array(channel))
#         obs['image'] = np.stack(resized_channels, axis=-1)
#         return obs