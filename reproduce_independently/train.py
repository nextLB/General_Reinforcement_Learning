

import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')

from reproduce_independently.agent.DQNAgent import DQNAgent
import torch

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    environment = "CarRacing-v3"
    maxEpisodes = 500
    learningRate = 0.0001
    gamma = 0.95
    startExplorationRate = 1.0
    endExplorationRate = 0.01
    explorationDecaySteps = 10000
    playBackBuffer = 30000
    batchSize = 32
    updateTargetNetworkFrequency = 50



def DQN_train():
    # 初始化DQN Agent
    DQNAgentInstance = DQNAgent(Config)

    episode = 0
    # 开始迭代训练
    for episode in range(Config.maxEpisodes):
        DQNAgentInstance.train_one_episode()


def main():
    DQN_train()


if __name__ == '__main__':
    main()




