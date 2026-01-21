

import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')

from reproduce_independently.agent.DQNAgent import DQNAgent


class Config:

    environment = "CarRacing-v3"
    maxEpisodes = 500
    frameStacks = 4
    learningRate = 0.0001
    gamma = 0.95
    startExplorationRate = 1.0
    endExplorationRate = 0.01
    playBackBuffer = 50000
    batchSize = 32
    updateTargetNetworkFrequency = 50



def DQN_train():
    # 初始化DQN Agent
    DQNAgentInstance = DQNAgent(Config)

    episode = 0
    # 开始迭代训练
    for episode in range(Config.maxEpisodes):
        action = DQNAgentInstance.select_action()
        print(action)


def main():
    DQN_train()


if __name__ == '__main__':
    main()




