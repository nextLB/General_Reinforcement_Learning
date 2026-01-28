

import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from dataclasses import dataclass
from reproduce_independently.agent.DQNAgent import DQNAgent
import torch
import os



class Config:
    version: str = "V1.1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # environment = "CarRacing-v3"
    environment = "PongNoFrameskip-v4"
    maxEpisodes = 1000
    frameStacks = 4
    learningRate = 1e-5
    gamma = 0.95
    startExplorationRate = 1.0
    endExplorationRate = 0.01
    explorationDecaySteps = 5000
    playBackBuffer = 50000
    batchSize = 8
    updateTargetNetworkFrequency = 50
    tau = 0.005
    saveModelEpisode = 20


@dataclass
class TrainingConfig:
    """训练配置参数"""
    version: str = "V1.2"
    environmentName: str = "PongNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageShape = (1, 120, 120)
    numActions = 0
    learningRate = 0.0001
    trainingEpisodes = 1000

    initialEpsilon = 1.0
    finalEpsilon = 0.1
    epsilonDecaySteps = 100000
    replayBufferCapacity = 30000
    discountFactor = 0.99
    targetUpdateFrequency = 1000000
    tau = 0.01
    max_grad_norm = 20.0
    lr_decay_steps = 10000



def DQN_train():
    # ============================================================= #
    # ============================================================= #
    # ============================================================= #
    #                       V1.1                                    #

    # # 初始化DQN Agent
    # DQNAgentInstance = DQNAgent(Config)
    # episode = 0
    # visualFlag = False
    # # 开始迭代训练
    # for episode in range(Config.maxEpisodes):
    #     # if (episode+1) % 10 == 0:
    #     #     visualFlag = True
    #     # else:
    #     #     visualFlag = False
    #     DQNAgentInstance.train_one_episode(visualFlag, episode)


    # ============================================================= #
    # ============================================================= #
    # ============================================================= #



    # ============================================================= #
    # ============================================================= #
    # ============================================================= #
    #                       V1.2                                    #
    os.makedirs(f'/home/next_lb/models/DQN_models/{TrainingConfig.environmentName}', exist_ok=True)
    # 创建配置类
    TrainingConfigInstance = TrainingConfig()

    # 设置内存优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化DQN Agent
    DQNAgentInstance = DQNAgent(TrainingConfigInstance)

    # 训练统计
    episodeRewards = []
    episodeLosses = []
    movingAverageRewards = []
    epsilonHistory = []
    bestAverageReward = -float('inf')
    for episode in range(TrainingConfigInstance.trainingEpisodes):
        episodeRewards, episodeLosses, movingAverageRewards, epsilonHistory, bestAverageReward = DQNAgentInstance.V12_train_one_episode(episode, episodeRewards, episodeLosses, movingAverageRewards, epsilonHistory, bestAverageReward)




    # ============================================================= #
    # ============================================================= #
    # ============================================================= #











def SAC_train():
    """
    Soft Actor-Critic(SAC)是一种基于策略梯度的深度强化学习算法，它具有最大化奖励与最大化熵(探索性)的双重目标。
    SAC通过引入熵正则项，使策略在决策时具有更大的随机性，从而提高探索能力。

    基于伪代码的实现与说明如下

        # 定义 SAC 超参数
        alpha = 0.2         # 熵正则项系数
        gamma = 0.99        # 折扣因子
        tau = 0.005         # 目标网络软更新参数
        lr = 3e-4           # 学习率

        # 初始化 Actor、Critic、Target Critic 网络和优化器
        actor = ActorNetwork()          # 策略网络  π(s)
        critic1 = CriticNetwork()       # 第一个 Q 网络 Q1(s, a)
        critic2 = CriticNetwork()       # 第二个 Q 网络 Q2(s, a)
        target_critic1 = CriticNetwork()    # 目标 Q 网络 1
        target_critic2 = CriticNetwork()    # 目标 Q 网络 2

        # 将目标 Q 网络的参数设置为与 Critic 网络相同
        target_critic1.load_state_dict(critic1.state_dict())
        target_critic2.load_state_dict(critic2.state_dict())

        # 初始化优化器
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=lr)
        critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=lr)

        # 经验回放池(Replay Buffer)
        replay_buffer = ReplayBuffer()

        # SAC 训练循环
        for each iteration:
            # Step 1: 从 Replay Buffer 中采样一个批次 (state, action, reward, next_state)
            batch = replay_buffer.sample()
            state, action, reward, next_state, done = batch

            # Step 2: 计算目标 Q 值(y)
            with torch.no_grad():
                # 从 Actor 网络中获取 next_state 的下一个动作
                next_action, next_log_prob = actor.sample(next_state)

                # 目标 Q 值的计算：使用目标 Q 网络的最小值 + 熵项
                target_q1_value = target_critic1(next_state, next_action)
                target_q2_value = target_critic2(next_state, next_action)
                min_target_q_value = torch.min(target_q1_value, target_q2_value)

                # 目标 Q 值 y = r + Y * (最小目标)






    """




    pass



def main():
    # DQN_train()
    SAC_train()


if __name__ == '__main__':
    main()




