#
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# from typing import Optional, Tuple
#
# from envs.car_racing import CarRacingEnvironment, Experience
# from network.DQN import DQNNetwork
#
#
# class DQNAgent:
#     """深度Q学习智能体"""
#
#     def __init__(self, config):
#         self.config = config
#         self.environment = None
#         self.policyNetwork = None
#         self.targetNetwork = None
#         self.optimizer = None
#         self.criterion = None
#
#         # 训练参数
#         self.epsilon = config.epsilonStart
#         self.epsilonDecay = config.epsilonDecay
#         self.epsilonEnd = config.epsilonEnd
#         self.gamma = config.gamma
#         self.batchSize = config.batchSize
#
#         # 训练状态
#         self.stepCount = 0
#         self.episodeCount = 0
#         self.totalReward = 0
#         self.trainingLosses = []
#
#         # 获取设备对象
#         self.device = torch.device(config.device)
#
#         # 初始化
#         self._initializeEnvironment()
#         self._initializeNetworks()
#         self._initializeOptimizer()
#
#     def _initializeEnvironment(self):
#         """初始化环境"""
#         try:
#             self.environment = CarRacingEnvironment(self.config)
#             initialState, info = self.environment.initialize()
#             self.actionSpace = self.environment.getActionSpace()
#             print(f"环境初始化成功，动作空间大小: {self.actionSpace}")
#         except Exception as e:
#             raise RuntimeError(f"环境初始化失败: {e}")
#
#     def _initializeNetworks(self):
#         """初始化网络"""
#         inputShape = (self.config.channels, self.config.height, self.config.width)
#
#         try:
#             # 策略网络
#             self.policyNetwork = DQNNetwork(inputShape, self.actionSpace).to(self.device)
#
#             # 目标网络
#             self.targetNetwork = DQNNetwork(inputShape, self.actionSpace).to(self.device)
#             self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
#             self.targetNetwork.eval()
#
#             print(f"网络初始化成功，输入形状: {inputShape}")
#
#         except Exception as e:
#             raise RuntimeError(f"网络初始化失败: {e}")
#
#     def _initializeOptimizer(self):
#         """初始化优化器"""
#         self.optimizer = optim.Adam(
#             self.policyNetwork.parameters(),
#             lr=self.config.learningRate,
#             eps=1e-8
#         )
#
#         # 使用Huber损失，对异常值更鲁棒
#         self.criterion = nn.SmoothL1Loss()
#
#     def selectAction(self, state: np.ndarray, trainingMode: bool = True) -> int:
#         """使用epsilon-greedy策略选择动作"""
#         if trainingMode and random.random() < self.epsilon:
#             # 探索：随机选择动作
#             return random.randrange(self.actionSpace)
#         else:
#             # 利用：选择Q值最大的动作
#             return self._selectGreedyAction(state)
#
#     def _selectGreedyAction(self, state: np.ndarray) -> int:
#         """选择贪婪动作"""
#         with torch.no_grad():
#             stateTensor = self._preprocessState(state).to(self.device)
#             qValues = self.policyNetwork(stateTensor)
#             return qValues.argmax().item()
#
#     def _preprocessState(self, state: np.ndarray) -> torch.Tensor:
#         """预处理状态"""
#         if not isinstance(state, np.ndarray):
#             raise ValueError(f"状态应为numpy数组，实际为: {type(state)}")
#
#         # 复制数据避免修改原始状态
#         state = state.copy()
#
#         # 归一化
#         if state.dtype != np.float32:
#             state = state.astype(np.float32) / 255.0
#
#         # 调整维度顺序：HWC -> CHW
#         if len(state.shape) == 3 and state.shape[2] == 3:
#             state = np.transpose(state, (2, 0, 1))
#         elif len(state.shape) == 4 and state.shape[3] == 3:
#             state = np.transpose(state, (0, 3, 1, 2))
#
#         # 转换为张量
#         return torch.FloatTensor(state)
#
#     def updateEpsilon(self):
#         """更新探索率"""
#         self.epsilon = max(self.epsilonEnd, self.epsilon * self.epsilonDecay)
#
#     def trainStep(self) -> Optional[float]:
#         """执行单步训练"""
#         # 检查缓冲区是否有足够的数据
#         if self.environment.getBufferSize() < self.batchSize:
#             return None
#
#         try:
#             # 采样批次
#             batch = self.environment.sampleBatch(self.batchSize)
#             if batch is None:
#                 return None
#
#             states, actions, rewards, nextStates, dones = batch
#
#             # 转换为张量
#             statesTensor = self._preprocessState(states).to(self.device)
#             actionsTensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
#             rewardsTensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
#             nextStatesTensor = self._preprocessState(nextStates).to(self.device)
#             donesTensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
#
#             # 计算当前Q值
#             currentQValues = self.policyNetwork(statesTensor).gather(1, actionsTensor)
#
#             # 计算目标Q值
#             with torch.no_grad():
#                 nextQValues = self.targetNetwork(nextStatesTensor).max(1)[0].unsqueeze(1)
#                 targetQValues = rewardsTensor + (self.gamma * nextQValues * (1 - donesTensor))
#
#             # 计算损失
#             loss = self.criterion(currentQValues, targetQValues)
#
#             # 反向传播
#             self.optimizer.zero_grad()
#             loss.backward()
#
#             # 梯度裁剪
#             nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), max_norm=1.0)
#
#             self.optimizer.step()
#
#             # 更新目标网络
#             self._updateTargetNetwork()
#
#             return loss.item()
#
#         except Exception as e:
#             print(f"训练步骤失败: {e}")
#             return None
#
#     def _updateTargetNetwork(self):
#         """更新目标网络"""
#         self.stepCount += 1
#
#         # 软更新或硬更新
#         if self.config.tau > 0:
#             # 软更新
#             for targetParam, policyParam in zip(
#                     self.targetNetwork.parameters(),
#                     self.policyNetwork.parameters()
#             ):
#                 targetParam.data.copy_(
#                     self.config.tau * policyParam.data +
#                     (1 - self.config.tau) * targetParam.data
#                 )
#         elif self.stepCount % self.config.targetUpdateFrequency == 0:
#             # 硬更新
#             self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
#
#     def trainEpisode(self) -> Tuple[float, Optional[float]]:
#         """训练一个完整回合"""
#         state, info = self.environment.reset()
#         episodeReward = 0
#         episodeLosses = []
#         done = False
#
#         while not done:
#             # 选择动作
#             action = self.selectAction(state)
#
#             # 执行动作
#             nextState, reward, terminated, truncated, info = self.environment.step(action)
#             done = terminated or truncated
#
#             # 存储经验
#             experience = Experience(state, action, reward, nextState, done)
#             self.environment.storeExperience(experience)
#
#             # 更新状态
#             state = nextState
#             episodeReward += reward
#
#             # 训练模型
#             loss = self.trainStep()
#             if loss is not None:
#                 episodeLosses.append(loss)
#
#         # 更新探索率
#         self.updateEpsilon()
#         self.episodeCount += 1
#
#         # 计算平均损失
#         avgLoss = np.mean(episodeLosses) if episodeLosses else None
#
#         return float(episodeReward), float(avgLoss) if avgLoss is not None else None
#
#     def validate(self, numEpisodes: int = 1) -> float:
#         """验证模型性能"""
#         originalEpsilon = self.epsilon
#         self.epsilon = 0.0  # 验证时使用纯贪婪策略
#
#         totalValidationReward = 0
#
#         for _ in range(numEpisodes):
#             state, info = self.environment.reset()
#             episodeReward = 0
#             done = False
#
#             while not done:
#                 action = self.selectAction(state, trainingMode=False)
#                 state, reward, terminated, truncated, info = self.environment.step(action)
#                 done = terminated or truncated
#                 episodeReward += reward
#
#             totalValidationReward += episodeReward
#
#         # 恢复探索率
#         self.epsilon = originalEpsilon
#
#         return float(totalValidationReward / numEpisodes)
#
#     def saveModel(self, filePath: str):
#         """保存模型"""
#         checkpoint = {
#             'policy_network_state_dict': self.policyNetwork.state_dict(),
#             'target_network_state_dict': self.targetNetwork.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epsilon': float(self.epsilon),
#             'step_count': self.stepCount,
#             'episode_count': self.episodeCount,
#             'config': self.config.toDict()
#         }
#
#         torch.save(checkpoint, filePath)
#         print(f"模型已保存到: {filePath}")
#
#     def loadModel(self, filePath: str):
#         """加载模型"""
#         try:
#             checkpoint = torch.load(filePath, map_location=self.device)
#
#             self.policyNetwork.load_state_dict(checkpoint['policy_network_state_dict'])
#             self.targetNetwork.load_state_dict(checkpoint['target_network_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#             self.epsilon = checkpoint['epsilon']
#             self.stepCount = checkpoint['step_count']
#             self.episodeCount = checkpoint.get('episode_count', 0)
#
#             print(f"模型已从 {filePath} 加载")
#
#         except Exception as e:
#             raise RuntimeError(f"加载模型失败: {e}")
#
#     def getBufferSize(self) -> int:
#         """获取经验缓冲区大小"""
#         return self.environment.getBufferSize()
#
#     def getTrainingStats(self) -> dict:
#         """获取训练统计信息"""
#         return {
#             'episode_count': self.episodeCount,
#             'step_count': self.stepCount,
#             'epsilon': float(self.epsilon),
#             'buffer_size': self.getBufferSize(),
#             'average_loss': float(np.mean(self.trainingLosses)) if self.trainingLosses else 0
#         }
#
#     def close(self):
#         """关闭环境"""
#         if self.environment:
#             self.environment.close()
#
#     def __del__(self):
#         """析构函数，确保资源释放"""
#         self.close()
#
#
