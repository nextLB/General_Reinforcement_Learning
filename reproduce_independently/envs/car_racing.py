
import copy
import gymnasium as gym
from collections import deque
import numpy as np

# 环境类
class CarRacingEnv:
    def __init__(self, frameStacks):
        self.environment = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=False
        )
        self.frameStacks = frameStacks
        self.frameBuffer = deque(maxlen=self.frameStacks)

        self.state = None
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.info = None

    def reset(self):
        state, info = self.environment.reset()  # 返回初始状态和info
        self.state = copy.deepcopy(state)
        self.info = copy.deepcopy(info)
        return state, info

    def step(self, action):
        nextState, reward, terminated, truncated, info = self.environment.step(action)
        self.state = copy.deepcopy(nextState)
        self.reward = copy.deepcopy(reward)
        self.terminated = copy.deepcopy(terminated)
        self.truncated = copy.deepcopy(truncated)
        self.info = copy.deepcopy(info)
        return nextState, reward, terminated, truncated, info

    # 进行帧堆叠
    def stack_frames(self):
        # 如果帧数不足，就用最后一帧补全
        if len(self.frameBuffer) != self.frameStacks:
            tempFramesList = list(self.frameBuffer)
            while len(tempFramesList) < self.frameStacks:
                tempFramesList.append(tempFramesList[-1] if tempFramesList else np.zeros_like(tempFramesList[0]))
            return np.concatenate(tempFramesList, axis=-1)
        else:
            return np.concatenate(self.frameBuffer, axis=-1)

    # 获取其动作空间大小
    def get_action_space(self):
        return self.environment.action_space.n



# 经验缓冲区类
class CarRacingExperienceBuffer:
    def __init__(self):
        pass


