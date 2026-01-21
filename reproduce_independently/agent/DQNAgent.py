import random
import sys
import copy
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.envs.car_racing import CarRacingEnv, CarRacingExperienceBuffer
from reproduce_independently.network.DQN import DQNNetWork





class DQNAgent:
    def __init__(self, config):
        self.config = config
        # 初始化环境实例与相应的经验缓冲区实例
        if self.config.environment == "CarRacing-v3":
            self.env = CarRacingEnv(self.config.frameStacks)
            self.experience = CarRacingExperienceBuffer()
            state, info = self.env.reset()
        else:
            self.env = CarRacingEnv(self.config.frameStacks)
            self.experience = CarRacingExperienceBuffer()
            state, info = self.env.reset()

        # 原始通道形状
        self.state = copy.deepcopy(state)
        self.imageShape = state.shape
        # 灰度图通道形状
        self.grayImageShape = (state.shape[0], state[1], 1)

        # 进行动作的相关信息初始化
        self.actionSpaceNumber = self.env.get_action_space()
        self.action = -1


        # 初始化策略网络
        self.policyNetwork = DQNNetWork(self.grayImageShape)
        # 初始化目标网络
        self.targetNetwork = DQNNetWork(self.grayImageShape)



    def select_action(self):
        self.action = random.randint(0, self.actionSpaceNumber - 1)
        return self.action


