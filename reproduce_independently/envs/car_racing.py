
import copy
import gymnasium as gym
from collections import deque
import numpy as np
import random

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
        # 将帧加入队列
        self.frameBuffer.append(self.state)
        return state, info

    def step(self, action):
        nextState, reward, terminated, truncated, info = self.environment.step(action)
        self.state = copy.deepcopy(nextState)
        self.reward = copy.deepcopy(reward)
        self.terminated = copy.deepcopy(terminated)
        self.truncated = copy.deepcopy(truncated)
        self.info = copy.deepcopy(info)
        # 将帧加入队列
        self.frameBuffer.append(self.state)
        return nextState, reward, terminated, truncated, info

    # 进行帧堆叠
    def stack_frames(self):
        # 如果帧数不足，就用最后一帧补全
        if len(self.frameBuffer) != self.frameStacks:
            while len(self.frameBuffer) < self.frameStacks:
                self.frameBuffer.append(self.frameBuffer[-1] if self.frameBuffer else np.zeros_like(self.frameBuffer[0]))
            return self.frameBuffer
        else:
            return self.frameBuffer

    # 获取其动作空间大小
    def get_action_space(self):
        # 关于CarRacing-v3的动作空间解析
        # 0 无操作
        # 1 左转(也可能是右转)
        # 2 右转(也可能是左转)
        # 3 踩油门(加速)
        # 4 踩刹车(减速)

        return self.environment.action_space.n

    # 处理state为灰度图
    def preprocess_state_to_gray(self, state):

        grayState = state
        # 如果状态是RGB图像，使用标准灰度转换公式
        if len(state.shape) == 3 and state.shape[-1] == 3:      # [H, W, 3]
            # 使用标准灰度转换公式: Y = 0.299R + 0.587G + 0.114B
            grayState = 0.299 * state[:, :, 0] + 0.587 * state[:, :, 1] + 0.114 * state[:, :, 2]
        elif len(state.shape) == 4 and state.shape[-1] == 3:    # [B, H, W, 3]
            grayState = 0.299 * state[:, :, :, 0] + 0.587 * state[:, :, :, 1] + 0.114 * state[:, :, :, 2]

        if len(grayState.shape) == 2:
            grayState = np.expand_dims(grayState, axis=0)

        return grayState.astype(state.dtype)



# 经验缓冲区类
class CarRacingExperienceBuffer:
    def __init__(self, playBackBuffer):
        self.playBackBufferSize = playBackBuffer
        self.experienceBuffer = deque(maxlen=self.playBackBufferSize)

    # 将经验存放入经验池
    def push(self, state, nextState, reward, terminated, truncated, info, action, done):
        self.experienceBuffer.append((state, nextState, reward, terminated, truncated, info, action, done))

    def get_current_buffer_size(self):
        return len(self.experienceBuffer)

    def sample_experience(self, batchSize):
        # 随机索引选取
        indices = random.sample(range(len(self.experienceBuffer)), batchSize)
        # 提取批次数据
        batch = [self.experienceBuffer[i] for i in indices]
        # 分离数据
        states = np.array([exp[0] for exp in batch])
        nextStates = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        actions = np.array([exp[6] for exp in batch])
        dones = np.array([exp[7] for exp in batch])

        return states, nextStates, actions, rewards, dones