import random
import sys
import copy
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.envs.car_racing import CarRacingEnv, CarRacingExperienceBuffer
from reproduce_independently.network.DQN import DQNNetWork
import matplotlib.pyplot as plt
import math
import torch





class DQNAgent:
    def __init__(self, config):
        self.config = config
        # 初始化环境实例与相应的经验缓冲区实例
        if self.config.environment == "CarRacing-v3":
            self.env = CarRacingEnv(self.config.batchSize)
            self.experience = CarRacingExperienceBuffer(self.config.playBackBuffer)
            state, info = self.env.reset()
        else:
            self.env = CarRacingEnv(self.config.batchSize)
            self.experience = CarRacingExperienceBuffer(self.config.playBackBuffer)
            state, info = self.env.reset()

        # 原始通道形状
        self.state = copy.deepcopy(state)
        self.imageShape = state.shape
        # 灰度图通道形状
        self.grayImageShape = (state.shape[0], state.shape[1], 1)

        # 进行动作的相关信息初始化
        self.actionSpaceNumber = self.env.get_action_space()
        self.action = -1


        # 初始化策略网络
        self.policyNetwork = DQNNetWork(self.grayImageShape, self.actionSpaceNumber).to(self.config.device)
        # 初始化目标网络
        self.targetNetwork = DQNNetWork(self.grayImageShape, self.actionSpaceNumber).to(self.config.device)

        self.stepCount = 0



    def select_action(self):

        # 计算当前ε值（指数衰减）
        epsThreshold = self.config.startExplorationRate + (self.config.startExplorationRate - self.config.endExplorationRate) * math.exp(-1. * self.stepCount / self.config.explorationDecaySteps)

        # 以ε概率随机选择动作，否则选择最优动作
        if random.random() > epsThreshold:
            grayState = self.env.preprocess_state_to_gray(self.state)
            grayStateTensor = torch.FloatTensor(grayState).to(self.config.device).unsqueeze(0)
            # 获取其Q值
            qValues = self.policyNetwork(grayStateTensor)
            # 选择最大Q值对应的动作
            self.action = qValues.max(1)[1].item()
        # 随机选择动作
        else:
            self.action = random.randint(0, self.actionSpaceNumber - 1)

        self.stepCount += 1
        return self.action


    def train_one_episode(self):
        state, info = self.env.reset()
        done = False

        # 初始化画布
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        while not done:

            # 检查是否有键盘事件
            if plt.get_fignums():  # 检查图形是否还存在
                try:
                    # 检查是否按下了Q键
                    if plt.waitforbuttonpress(0.001):
                        key = plt.gcf().canvas.key_press_event.key
                        if key == 'q':
                            print("Q pressed: Stopping episode")
                            break
                except:
                    pass  # 忽略事件处理中的异常

            # 获取动作
            action = self.select_action()
            # 作用于环境并获取返回结果
            nextState, reward, terminated, truncated, info = self.env.step(action)
            self.state = copy.deepcopy(nextState)
            # 存放如经验区
            self.experience.push(nextState, reward, terminated, truncated, info)

            # 可视化图像
            im = ax.imshow(nextState)
            ax.set_title('State')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.axis('off')
            plt.draw()
            plt.pause(0.001)

            # 判断本回合是否结束
            done = terminated or truncated

        # 关闭图形窗口
        plt.close(fig)
        plt.ioff()

        self.env.frameBuffer.clear()

