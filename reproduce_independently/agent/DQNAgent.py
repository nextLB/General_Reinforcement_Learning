
import torch
from reproduce_independently.envs.car_racing import CarRacingEnvironment


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


    # 初始化环境
    def __initializeEnvironment(self):
        pass








