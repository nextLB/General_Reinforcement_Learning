

from envs.car_racing import CarRacingEnv
from agent.DQNAgent import DQNAgent


class Config:
    version = 'V1'






def train():

    Agent = DQNAgent(Config.version)
    print(Agent)



def main():
    train()





if __name__ == '__main__':
    main()


