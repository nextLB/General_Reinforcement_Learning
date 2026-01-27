

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







    # ============================================================= #
    # ============================================================= #
    # ============================================================= #


def main():
    DQN_train()



if __name__ == '__main__':
    main()




