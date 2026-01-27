



import numpy as np
from PIL import Image
import gymnasium as gym


class PNFSV4Environment:
    def __init__(self, config):
        self.environmentName = "PongNoFrameskip-v4"
        self.config = config
        self.env = gym.make(self.config.environmentName, render_mode='rgb_array')

