# # car_racing.py
# import gymnasium as gym
# import random
# import numpy as np
# from collections import deque, namedtuple
# from typing import Optional, List, Tuple
#
# # 定义经验元组
# Experience = namedtuple('Experience',
#     ['state', 'action', 'reward', 'nextState', 'done'])
#
# class CarRacingEnvironment:
#     """Car Racing游戏环境封装类"""
#
#     def __init__(self, config):
#         self.config = config
#         self.env = None
#         self.replayBuffer = None
#         self.bufferSize = 0
#         self.currentState = None
#         self.isInitialized = False
#
#     def initialize(self):
#         """初始化环境"""
#         try:
#             self.env = gym.make(
#                 self.config.environmentName,
#                 render_mode=self.config.renderMode,
#                 lap_complete_percent=self.config.lapCompletePercent,
#                 domain_randomize=self.config.domainRandomize,
#                 continuous=self.config.continuous
#             )
#
#             self.currentState, info = self.env.reset()
#             self.replayBuffer = deque(maxlen=self.config.replayBufferCapacity)
#             self.isInitialized = True
#
#             # 更新配置中的图像尺寸
#             self.config.height = self.currentState.shape[0]
#             self.config.width = self.currentState.shape[1]
#             self.config.channels = self.currentState.shape[2]
#
#             return self.currentState, info
#
#         except Exception as e:
#             raise RuntimeError(f"环境初始化失败: {e}")
#
#     def reset(self) -> Tuple[np.ndarray, dict]:
#         """重置环境"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#
#         try:
#             self.currentState, info = self.env.reset()
#             return self.currentState, info
#         except Exception as e:
#             raise RuntimeError(f"环境重置失败: {e}")
#
#     def getActionSpace(self) -> int:
#         """获取动作空间大小"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#         return self.env.action_space.n
#
#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
#         """执行动作"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#
#         try:
#             nextState, reward, terminated, truncated, info = self.env.step(action)
#             done = terminated or truncated
#             self.currentState = nextState
#             return nextState, reward, terminated, truncated, info
#         except Exception as e:
#             raise RuntimeError(f"执行动作失败: {e}")
#
#     def render(self):
#         """渲染环境（用于可视化）"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#         return self.env.render()
#
#     def close(self):
#         """关闭环境"""
#         if self.env:
#             self.env.close()
#             self.isInitialized = False
#
#     def storeExperience(self, experience: Experience):
#         """存储经验到回放缓冲区"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#
#         self.replayBuffer.append(experience)
#         self.bufferSize = len(self.replayBuffer)
#
#     def sampleBatch(self, batchSize: int) -> Optional[Tuple]:
#         """从回放缓冲区采样批次"""
#         if not self.isInitialized:
#             raise RuntimeError("环境未初始化")
#
#         if self.bufferSize < batchSize:
#             return None
#
#         try:
#             batch = random.sample(self.replayBuffer, batchSize)
#
#             # 解包批次数据
#             states = np.array([exp.state for exp in batch], dtype=np.float32)
#             actions = np.array([exp.action for exp in batch], dtype=np.int64)
#             rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
#             nextStates = np.array([exp.nextState for exp in batch], dtype=np.float32)
#             dones = np.array([exp.done for exp in batch], dtype=np.float32)
#
#             return states, actions, rewards, nextStates, dones
#
#         except Exception as e:
#             print(f"批次采样失败: {e}")
#             return None
#
#     def getBufferSize(self) -> int:
#         """获取当前缓冲区大小"""
#         return self.bufferSize
#
#     def getCurrentState(self) -> np.ndarray:
#         """获取当前状态"""
#         return self.currentState
#
#     def preprocessState(self, state: np.ndarray) -> np.ndarray:
#         """预处理状态（可选，如归一化）"""
#         if state is None:
#             return None
#
#         # 简单归一化到[0, 1]
#         processed = state.astype(np.float32) / 255.0
#
#         # 调整维度顺序：HWC -> CHW
#         if len(processed.shape) == 3 and processed.shape[2] == 3:
#             processed = np.transpose(processed, (2, 0, 1))
#
#         return processed
#
#     def getStateShape(self) -> tuple:
#         """获取状态形状"""
#         if not self.isInitialized:
#             return None
#         return self.currentState.shape
#
#     def getNumActions(self) -> int:
#         """获取动作数量"""
#         if not self.isInitialized:
#             return 0
#         return self.env.action_space.n
#
#     def validateAction(self, action: int) -> bool:
#         """验证动作是否有效"""
#         if not self.isInitialized:
#             return False
#         return 0 <= action < self.env.action_space.n
#
# class ExperienceReplayBuffer:
#     """经验回放缓冲区（独立版本，可选使用）"""
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#         self.size = 0
#
#     def add(self, experience: Experience):
#         """添加经验"""
#         self.buffer.append(experience)
#         self.size = len(self.buffer)
#
#     def sample(self, batchSize: int) -> Optional[Tuple]:
#         """采样批次"""
#         if self.size < batchSize:
#             return None
#
#         batch = random.sample(self.buffer, batchSize)
#
#         states = np.array([exp.state for exp in batch], dtype=np.float32)
#         actions = np.array([exp.action for exp in batch], dtype=np.int64)
#         rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
#         nextStates = np.array([exp.nextState for exp in batch], dtype=np.float32)
#         dones = np.array([exp.done for exp in batch], dtype=np.float32)
#
#         return states, actions, rewards, nextStates, dones
#
#     def clear(self):
#         """清空缓冲区"""
#         self.buffer.clear()
#         self.size = 0
#
#     def getSize(self) -> int:
#         """获取缓冲区大小"""
#         return self.size
#
#     def isFull(self) -> bool:
#         """检查缓冲区是否已满"""
#         return self.size >= self.capacity