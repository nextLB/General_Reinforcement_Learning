import random
import sys
import copy
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.envs.car_racing import CarRacingEnv, CarRacingExperienceBuffer
from reproduce_independently.network.DQN import DQNNetWork
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
import torch.nn as nn



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
        self.targetNetwork.eval()
        # 定义优化器
        self.optimizer = optim.Adam(
            self.policyNetwork.parameters(),
            lr=self.config.learningRate
        )
        # 定义损失函数
        self.lossFn = nn.SmoothL1Loss()


        self.stepCount = 0



    def select_action(self):

        # 计算当前ε值（指数衰减）
        epsThreshold = self.config.endExplorationRate + (self.config.startExplorationRate - self.config.endExplorationRate) * math.exp(-1. * self.stepCount / self.config.explorationDecaySteps)

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

    def soft_update_target_network(self):
        for targetParam, policyParam in zip(self.targetNetwork.parameters(), self.policyNetwork.parameters()):
            targetParam.data.copy_(self.config.tau * policyParam.data + (1.0 - self.config.tau) * targetParam.data)



    def train_one_episode(self, visualFlag, episode):
        self.stepCount = 0
        state, info = self.env.reset()
        done = False
        averageLoss = 0
        averageReward = 0

        if visualFlag:
            # 初始化画布
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        while not done:

            # 获取动作
            action = self.select_action()
            # 作用于环境并获取返回结果
            nextState, reward, terminated, truncated, info = self.env.step(action)
            averageReward += reward
            averageReward /= self.stepCount

            # 存放经验到缓冲区
            self.experience.push(self.state, nextState, reward, terminated, truncated, info, action, done)
            self.state = copy.deepcopy(nextState)

            # 接下来如果符合条件的话，进行模型的更新
            if self.experience.get_current_buffer_size() >= self.config.playBackBuffer:
                states, nextStates, actions, rewards, dones = self.experience.sample_experience(self.config.batchSize)
                # 转换为灰度图，以通过模型网络
                stateTensors = []
                nextStateTensors = []
                for i in range(len(states)):
                    stateTensors.append(self.env.preprocess_state_to_gray(states[i]))
                    nextStateTensors.append(self.env.preprocess_state_to_gray(nextStates[i]))

                stateTensors = torch.FloatTensor(stateTensors).to(self.config.device)
                nextStateTensors = torch.FloatTensor(nextStateTensors).to(self.config.device)
                rewardTensors = torch.FloatTensor(rewards).to(self.config.device).unsqueeze(1)
                actionTensors = torch.LongTensor(actions).to(self.config.device).unsqueeze(1)
                doneTensors = torch.FloatTensor(dones).to(self.config.device).unsqueeze(1)

                # 计算当前Q值
                currentQValues = self.policyNetwork(stateTensors).gather(1, actionTensors)

                # 计算目标的Q值
                with torch.no_grad():
                    # 使用策略网络选择动作
                    nextActions = self.policyNetwork(nextStateTensors).argmax(1).unsqueeze(1)
                    # 计算下一个状态的最大Q值
                    nextQValues = self.targetNetwork(nextStateTensors).gather(1, nextActions)
                    # 计算目标Q值
                    targetQValues = rewardTensors + (self.config.gamma * nextQValues * (1 - doneTensors))

                # 计算损失  目前是Huber loss
                loss = self.lossFn(currentQValues, targetQValues)

                # 反向传播优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), max_norm=10.0)

                self.optimizer.step()

                averageLoss += loss.item()
                averageLoss /= self.stepCount

                # 对目标网络进行软更新
                self.soft_update_target_network()

            # 判断本回合是否结束
            done = terminated or truncated


            if self.stepCount % 100 == 0:
                print(f'episode: {episode}, self.stepCount: {self.stepCount}, averageLoss: {averageLoss}, averageReward: {averageReward}, self.experience.get_current_buffer_size(): {self.experience.get_current_buffer_size()}')


            if visualFlag:
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

                # 可视化图像
                im = ax.imshow(nextState)
                ax.set_title('State')
                ax.set_xlabel('Width')
                ax.set_ylabel('Height')
                ax.axis('off')
                plt.draw()
                plt.pause(0.001)

        if visualFlag:
            # 关闭图形窗口
            plt.close(fig)
            plt.ioff()

        self.env.frameBuffer.clear()

