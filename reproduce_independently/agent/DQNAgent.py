import random
import sys
import copy
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.envs.car_racing import CarRacingEnv, CarRacingExperienceBuffer
from reproduce_independently.envs.pong_no_frameskip import PNFSV4Environment, PongExperienceBuffer
from reproduce_independently.network.DQN import DQNNetWork, ResNetDeepQNetwork
import matplotlib.pyplot as plt
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np



class DQNAgent:
    def __init__(self, config):
        self.config = config

        if self.config.version == "V1.2":
            if self.config.environmentName == "PongNoFrameskip-v4":
                self.env = PNFSV4Environment(self.config)
                # 初始化经验池
                self.Experience = PongExperienceBuffer(self.config.replayBufferCapacity)
            else:
                print('No Do No Die!')
            self.config.numActions = self.env.actionSpace.n
            self.policyNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions).to(
                self.config.device)
            self.targetNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions).to(
                self.config.device)

            self._updateTargetNetwork()
            self.targetNetwork.eval()

            # 优化器
            self.optimizer = optim.Adam(
                self.policyNetwork.parameters(),
                lr=self.config.learningRate,
                eps=1e-4,
                weight_decay=1e-5  # 添加L2正则化
            )
            # 学习率调度器
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps,
                gamma=0.5
            )
            # 训练状态
            self.stepsCompleted = 0
            self.episodesCompleted = 0

            # 用于Double DQN
            self.last_loss = 0.0



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

            return



    # ============================================================= #
    # ============================================================= #
    # ============================================================= #
    #                       V1.1                                    #


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

    # ============================================================= #
    # ============================================================= #
    # ============================================================= #



    # ============================================================= #
    # ============================================================= #
    # ============================================================= #
    #                       V1.2                                    #

    def _updateTargetNetwork(self) -> None:
        """更新目标网络参数"""
        # 硬更新
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

        # 或者使用软更新（更稳定）
        target_net_state_dict = self.targetNetwork.state_dict()
        policy_net_state_dict = self.policyNetwork.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.tau + \
                                         target_net_state_dict[key] * (1 - self.config.tau)
        self.targetNetwork.load_state_dict(target_net_state_dict)



    def selectAction(self, state):
        randomValue = random.random()
        epsilon = self._calculateCurrentEpsilon()

        self.stepsCompleted += 1
        if randomValue > epsilon:
            with torch.no_grad():
                qValues = self.policyNetwork(state)
                return qValues.max(1)[1].item()
        else:
            return random.randrange(self.config.numActions)

    def _calculateCurrentEpsilon(self) -> float:
        """计算当前的epsilon值"""
        return self.config.finalEpsilon + (self.config.initialEpsilon - self.config.finalEpsilon) * \
            np.exp(-1.0 * self.stepsCompleted / self.config.epsilonDecaySteps)

    def getCurrentEpsilon(self) -> float:
        """获取当前epsilon值"""
        return self._calculateCurrentEpsilon()

    def optimizeModel(self, experience):
        # 采样
        states, actions, rewards, next_states, dones = experience.sample(16)

        # 2. 确保数据在正确的设备和数据类型上
        states = torch.as_tensor(states, dtype=torch.float32, device=self.config.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.config.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.config.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.config.device)

        # 3. 实现Double DQN（减少Q值高估）
        with torch.no_grad():
            # 使用policy网络选择动作
            next_actions = self.policyNetwork(next_states).max(1)[1]
            # 使用target网络评估Q值
            next_q_values = self.targetNetwork(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

            # 计算目标Q值
            target_q_values = rewards + (self.config.discountFactor * next_q_values * (1 - dones))

        # 4. 计算当前Q值
        current_q_values = self.policyNetwork(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 5. 计算损失 - 使用Huber loss提高稳定性
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # 6. 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 7. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.config.max_grad_norm)

        # 8. 优化步骤
        self.optimizer.step()

        # 9. 定期更新目标网络
        if self.stepsCompleted % self.config.targetUpdateFrequency == 0:
            self._updateTargetNetwork()

        # 10. 更新学习率
        if self.stepsCompleted % self.config.lr_decay_steps == 0:
            self.scheduler.step()

        self.last_loss = loss.item()
        return loss.item()


    def getTrainingStatistics(self) -> dict:
        """获取训练统计信息"""
        return {
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
            'currentEpsilon': self.getCurrentEpsilon(),
        }

    def saveCheckpoint(self, filePath: str) -> None:
        """保存模型检查点"""
        checkpoint = {
            'policyNetworkState': self.policyNetwork.state_dict(),
            'targetNetworkState': self.targetNetwork.state_dict(),
            'optimizerState': self.optimizer.state_dict(),
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
            'config': self.config
        }
        torch.save(checkpoint, filePath)


    def V12_train_one_episode(self, episode, episodeRewards, episodeLosses, movingAverageRewards, epsilonHistory, bestAverageReward):
        state, info = self.env.reset()
        totalReward = 0.0
        stepsInEpisode = 0
        totalLoss = 0.0
        lossCount = 0
        loss = 0

        while True:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
                state = torch.unsqueeze(state, 0)
                state = torch.unsqueeze(state, 0)
            state = state.to(self.config.device)
            action = self.selectAction(state)
            # 输入的环境中进行交互，返回信息
            nextState, reward, done, info = self.env.step(action)
            # 添加到经验池中
            self.Experience.push(state, action, reward, nextState, done)

            # 满足一定经验池的数量限制后再进行优化模型
            if len(self.Experience.buffer) >= self.config.replayBufferCapacity:
                # 优化模型
                loss = self.optimizeModel(self.Experience)

            # 统计信息与数据
            totalLoss += loss
            lossCount += 1
            state = nextState
            totalReward += reward
            stepsInEpisode += 1

            if done:
                break

        self.episodesCompleted += 1


        # 记录统计信息
        averageLoss = totalLoss / lossCount if lossCount > 0 else 0.0
        episodeRewards.append(totalReward)
        episodeLosses.append(averageLoss)
        epsilonHistory.append(self.getCurrentEpsilon())

        # 计算移动平均奖励  五十个回合内的
        if len(episodeRewards) >= 50:
            movingAverage = np.mean(episodeRewards[-50:])
        else:
            movingAverage = np.mean(episodeRewards)
        movingAverageRewards.append(movingAverage)

        # 保存更新最佳模型
        if movingAverage > bestAverageReward:
            bestAverageReward = movingAverage
            self.saveCheckpoint(
                f"/home/next_lb/models/DQN_models/{self.config.environmentName}/best_model.pth"
            )

        # 定期日志输出
        if episode % 5 == 0:
            stats = self.getTrainingStatistics()
            print(
                f"回合 {episode:4d} | "
                f"奖励: {totalReward:7.2f} | "
                f"步数: {stepsInEpisode:4d} | "
                f"移动平均: {movingAverage:7.2f} | "
                f"平均损失: {averageLoss:7.4f} | "
                f"Epsilon: {stats['currentEpsilon']:.3f} | "
            )

        return episodeRewards, episodeLosses, movingAverageRewards, epsilonHistory, bestAverageReward


