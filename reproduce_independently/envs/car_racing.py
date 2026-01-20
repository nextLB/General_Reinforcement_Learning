
import gymnasium as gym
from collections import deque, namedtuple
import random
import numpy as np

# 定义经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])


class CarRacingEnvironment:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.replayBuffer = None
        self.bufferSize = 0
        self.currentState = None
        self.isInitialized = False


    # 初始化环境
    def initialize(self):
        try:
            self.env = gym.make(
                self.config.environmentName,
                render_mode=self.config.renderMode,
                lap_complete_percent=self.config.lapCompletePercent,
                domain_randomize=self.config.domainRandomize,
                continuous=self.config.continuous
            )

            self.currentState, info = self.env.reset()
            self.replayBuffer = deque(maxlen=self.config.replayBufferCapacity)
            self.isInitialized = True

            # 更新配置中的图像尺寸
            self.config.height = self.currentState.shape[0]
            self.config.width = self.currentState.shape[1]
            # self.config.channels = self.currentState.shape[2]
            # 使用灰度图处理
            self.config.channels = 1

            return self.currentState, info

        except Exception as e:
            raise RuntimeError(f"环境初始化失败：{e}")


    # 重置环境
    def reset(self):
        if not self.isInitialized:
            raise RuntimeError('环境未初始化')

        try:
            self.currentState, info = self.env.reset()
            return self.currentState, info
        except Exception as e:
            raise RuntimeError(f"环境重置失败: {e}")

    # 获取动作空间大小
    def getActionSpace(self):
        if not self.isInitialized:
            raise RuntimeError('环境未初始化')
        return self.env.action_space.n

    # 执行动作
    def step(self, action):
        if not self.isInitialized:
            raise RuntimeError("环境未初始化")

        try:
            nextState, reward, terminated, truncated, info = self.env.step(action)
            # 奖励裁剪和缩放
            if reward < 0:
                reward = reward * 0.5  # 轻微惩罚
            else:
                reward = reward * 2.0  # 增强正奖励

            reward = np.clip(reward, -1.0, 1.0)  # 更严格的裁剪

            self.currentState = nextState
            return nextState, reward, terminated, truncated, info
        except Exception as e:
            raise RuntimeError(f"执行动作失败: {e}")


    # 渲染环境  用于可视化
    def render(self):
        if not self.isInitialized:
            raise RuntimeError("环境初始化")
        return self.env.render()

    # 关闭环境
    def close(self):
        if self.env:
            self.env.close()
            self.isInitialized = False


    # 存储经验到回放缓冲区
    def storeExperience(self, experience):
        if not self.isInitialized:
            raise RuntimeError("环境未初始化")

        self.replayBuffer.append(experience)
        self.bufferSize = len(self.replayBuffer)


    # 从回放缓冲区采样批次
    def sampleBatch(self, batchSize):
        if not self.isInitialized:
            raise RuntimeError("环境未初始化")

        if self.bufferSize < batchSize:
            return None

        try:
            batch = random.sample(self.replayBuffer, batchSize)

            # 解包批次数据
            states = np.array([exp.state for exp in batch], dtype=np.float32)
            actions = np.array([exp.action for exp in batch], dtype=np.int64)
            rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
            nextStates = np.array([exp.nextState for exp in batch], dtype=np.float32)
            dones = np.array([exp.done for exp in batch], dtype=np.float32)

            return states, actions, rewards, nextStates, dones

        except Exception as e:
            return None


    # 获取当前缓冲区大小
    def getBufferSize(self):
        return self.bufferSize



