import random
import sys
import copy
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.envs.car_racing import CarRacingEnv, CarRacingExperienceBuffer
from reproduce_independently.envs.pong_no_frameskip import PNFSV4Environment
from reproduce_independently.network.DQN import DQNNetWork
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os



class DQNAgent:
    def __init__(self, config):
        self.config = config

        if self.config.version == "V1.2":
            print('dadsadasdasdas')
            return
        else:
            # 初始化环境实例与相应的经验缓冲区实例
            if self.config.environment == "CarRacing-v3":
                self.env = CarRacingEnv(self.config.frameStacks)
                self.experience = CarRacingExperienceBuffer(self.config.playBackBuffer)
                state, info = self.env.reset()
            else:
                self.env = CarRacingEnv(self.config.frameStacks)
                self.experience = CarRacingExperienceBuffer(self.config.playBackBuffer)
                state, info = self.env.reset()

            # 原始通道形状
            self.imageShape = state.shape
            # 灰度图通道形状
            self.grayImageShape = (state.shape[0], state.shape[1], self.config.frameStacks)

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
            # 定义学习率调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',     # 监控loss下降
                factor=0.5,     # 每次降低为原来的一半
                patience=5,     # 连续5个episode loss不下降就降低
            )


            self.stepCount = 0

            self.stateFrames = []
            self.nextStateFrames = []



    def select_action(self):

        # 计算当前ε值（指数衰减）
        epsThreshold = self.config.endExplorationRate + (self.config.startExplorationRate - self.config.endExplorationRate) * math.exp(-1. * self.stepCount / self.config.explorationDecaySteps)

        # 以ε概率随机选择动作，否则选择最优动作
        if random.random() > epsThreshold:
            grayStates = self.env.stack_frames()
            grayList = []
            for i in range(len(grayStates)):
                grayList.append(self.env.preprocess_state_to_gray(grayStates[i]))
            grayStateTensors = torch.FloatTensor(grayList).to(self.config.device).unsqueeze(0)
            # 获取其Q值
            qValues = self.policyNetwork(grayStateTensors)
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

    def hard_update_target_network(self):
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def train_one_episode(self, visualFlag, episode):
        self.stepCount = 0
        state, info = self.env.reset()
        done = False
        totalLoss = 0
        totalReward = 0

        if visualFlag:
            # 初始化画布
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        while not done:
            # 获取当前的堆叠帧
            self.stateFrames = copy.deepcopy(self.env.stack_frames())
            # 获取动作
            action = self.select_action()
            # 作用于环境并获取返回结果
            nextState, reward, terminated, truncated, info = self.env.step(action)
            totalReward += reward
            # 获取作用过后的堆叠帧
            self.nextStateFrames = copy.deepcopy(self.env.stack_frames())

            # 存放经验到缓冲区
            self.experience.push(self.stateFrames, self.nextStateFrames, reward, terminated, truncated, info, action, done)

            # 接下来如果符合条件的话，进行模型的更新
            if self.experience.get_current_buffer_size() >= (self.config.playBackBuffer / 10):
                states, nextStates, actions, rewards, dones = self.experience.sample_experience(self.config.batchSize)
                # 转换为灰度图，以通过模型网络
                stateTensors = []
                nextStateTensors = []
                for i in range(len(states)):
                    tempStateList = []
                    tempNextStateList = []
                    for j in range(len(states[i])):
                        tempStateList.append(self.env.preprocess_state_to_gray(states[i][j]))
                        tempNextStateList.append(self.env.preprocess_state_to_gray(nextStates[i][j]))
                    stateTensors.append(copy.deepcopy(tempStateList))
                    nextStateTensors.append(copy.deepcopy(tempNextStateList))

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
                torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), max_norm=1.0)

                self.optimizer.step()

                totalLoss += loss.item()

                # 对目标网络进行软更新
                self.soft_update_target_network()

                # 符合条件进行硬更新
                if episode % self.config.updateTargetNetworkFrequency == 0:
                    self.hard_update_target_network()

            # 判断本回合是否结束
            done = terminated or truncated


            if self.stepCount % 1000 == 0:
                print(f'episode: {episode}, self.stepCount: {self.stepCount}, totalLoss: {totalLoss}, totalReward: {totalReward}, self.experience.get_current_buffer_size(): {self.experience.get_current_buffer_size()}')


            if visualFlag and self.stepCount % 100 == 0:
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
                plt.pause(0.00001)

        if visualFlag:
            # 关闭图形窗口
            plt.close(fig)
            plt.ioff()

        averageLoss = totalLoss / self.stepCount
        averageReward = totalReward / self.stepCount
        # 学习率的步入
        self.scheduler.step(averageLoss)

        print(f'averageLoss: {averageLoss}, averageReward: {averageReward}')

        self.env.frameBuffer.clear()

        # 判断下是否要保存模型参数文件下来
        if episode % self.config.saveModelEpisode == 0:
            checkpoint = {
                'episode': episode,
                'policy_network_state_dict': self.policyNetwork.state_dict(),
                'target_network_state_dict': self.targetNetwork.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step_count': self.stepCount,
            }

            os.makedirs(f'/home/next_lb/models/DQN_models/{self.config.environment}', exist_ok=True)
            torch.save(checkpoint, f'/home/next_lb/models/DQN_models/{self.config.environment}/checkpoint_episode.pth')
            print(f'模型已成功保存至: /home/next_lb/models/DQN_models/checkpoint_episode.pth')


