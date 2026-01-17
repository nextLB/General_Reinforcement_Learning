
import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
import torch
from reproduce_independently.envs.car_racing import CarRacingEnvironment
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.environment = None
        self.policyNetwork = None
        self.targetNetwork = None
        self.optimizer = None
        self.criterion = None

        # 训练参数
        self.epsilon = config.epsilonStart
        self.epsilonDecay = config.epsilonDecay
        self.epsilonEnd = config.epsilonEnd
        self.gamma = config.gamma
        self.batchSize = config.batchSize

        # 训练状态
        self.stepCount = 0
        self.episodeCount = 0
        self.totalReward = 0
        self.trainingLosses = []

        # 获取设备对象
        self.device = torch.device(config.device)

        # 初始化
        self._initializeEnvironment()
        self._initializeNetworks()
        self._initializeOptimizer()



    # 初始化环境
    def _initializeEnvironment(self):
        try:
            self.environment = CarRacingEnvironment(self.config)
            initialState, info = self.environment.initialize()
            self.actionSpace = self.environment.getActionSpace()
        except Exception as e:
            raise RuntimeError()


    # 初始化网络
    def _initializeNetworks(self):
        inputShape = (self.config.channels, self.config.height, self.config.width)
        try:
            # 策略网络
            pass
        except Exception as e:
            raise RuntimeError(f"网络初始化失败: {e}")


    # 初始化优化器
    def _initializeOptimizer(self):
        pass





