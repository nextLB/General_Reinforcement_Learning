

import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')

from reproduce_independently.agent.DQNAgent import DQNAgent
import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    environment = "CarRacing-v3"
    maxEpisodes = 1000
    frameStacks = 4
    learningRate = 0.00001
    gamma = 0.99
    startExplorationRate = 1.0
    endExplorationRate = 0.01
    explorationDecaySteps = 10000
    playBackBuffer = 30000
    batchSize = 16
    updateTargetNetworkFrequency = 50
    tau = 0.005
    saveModelEpisode = 20



def DQN_train():
    # 初始化DQN Agent
    DQNAgentInstance = DQNAgent(Config)

    episode = 0
    visualFlag = False
    # 开始迭代训练
    for episode in range(Config.maxEpisodes):
        # if (episode+1) % 10 == 0:
        #     visualFlag = True
        # else:
        #     visualFlag = False
        DQNAgentInstance.train_one_episode(visualFlag, episode)


def main():
    DQN_train()


if __name__ == '__main__':
    main()




