



import torch
import random
from collections import deque, namedtuple
import numpy as np
from PIL import Image
import gymnasium as gym


class PNFSV4Environment:
    def __init__(self, config):
        self.environmentName = "PongNoFrameskip-v4"
        self.config = config
        self.env = gym.make(self.config.environmentName, render_mode='rgb_array')


    # 对于Pong游戏的图像帧进行预处理
    def preprocess_frame_to_gray(self, frame):
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)  # 使用numpy提高效率

        # 调整大小
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.config.imageShape[1], self.config.imageShape[2]), Image.BILINEAR)
        frame = np.array(img)

        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    # 重置环境并返回预处理后的初始状态帧
    def reset(self):
        state, info = self.env.reset()
        processedState = self.preprocess_frame_to_gray(state)
        return processedState, info

    # 执行动作并返回预处理后的结果
    def step(self, action):
        nextState, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        processedNextState = self.preprocess_frame_to_gray(nextState)
        return processedNextState, reward, done, info

    @property
    def actionSpace(self):
        return self.env.action_space

    @property
    def observationSpace(self):
        return self.env.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.env.close()







Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])



class PongExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
        """添加经验到缓冲区"""
        # 确保状态在GPU上以加速训练
        nextState = torch.from_numpy(nextState)
        stateGpu = state.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        nextStateGpu = nextState.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.buffer.append(Experience(stateGpu, action, reward, nextStateGpu, done))

    def sample(self, batchSize: int) -> tuple:
        """从缓冲区中随机采样经验"""
        experiences = random.sample(self.buffer, batchSize)

        # 批量处理状态
        states = torch.cat([exp.state for exp in experiences], dim=0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        nextStates = torch.cat([exp.nextState for exp in experiences], dim=0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        nextStates = torch.unsqueeze(nextStates, 0)
        nextStates = torch.unsqueeze(nextStates, 0)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        return states, actions, rewards, nextStates, dones

    def __len__(self) -> int:
        return len(self.buffer)

