

import sys
sys.path.append('/home/next_lb/桌面/next/General_Reinforcement_Learning')
from reproduce_independently.train import Config
from reproduce_independently.envs.car_racing import CarRacingEnv
from reproduce_independently.network.DQN import DQNNetWork
from train import TrainingConfig
from reproduce_independently.network.DQN import ResNetDeepQNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from PIL import Image


def CarRacingDQNInference():
    # 初始化环境类
    CarRacingEnvInstance = CarRacingEnv(Config.frameStacks)
    state, info = CarRacingEnvInstance.reset()
    # 灰度图通道形状
    grayImageShape = (state.shape[0], state.shape[1], Config.frameStacks)
    actionSpaceNumber = CarRacingEnvInstance.get_action_space()
    # 初始化网络模型
    DQNNetWorkInstance = DQNNetWork(grayImageShape, actionSpaceNumber)
    checkpointPath = '/home/next_lb/models/DQN_models/checkpoint_episode.pth'
    checkpoint = torch.load(checkpointPath, map_location=Config.device)

    # 加载策略网络权重
    DQNNetWorkInstance.load_state_dict(checkpoint['policy_network_state_dict'])
    DQNNetWorkInstance.to(Config.device)
    DQNNetWorkInstance.eval()

    # 初始化画布
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    done = False
    while not done:
        frameSates = CarRacingEnvInstance.stack_frames()
        grayList = []
        for i in range(len(frameSates)):
            grayList.append(CarRacingEnvInstance.preprocess_state_to_gray(frameSates[i]))
        grayStateTensors = torch.FloatTensor(grayList).to(Config.device).unsqueeze(0)
        # 获取其Q值
        qValues = DQNNetWorkInstance(grayStateTensors)
        # 选择最大Q值对应的动作
        action = qValues.max(1)[1].item()
        # 作用于环境并获取返回结果
        nextState, reward, terminated, truncated, info = CarRacingEnvInstance.step(action)

        # 判断本回合是否结束
        done = terminated or truncated

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
        print(f'action: {action}')

    # 关闭图形窗口
    plt.close(fig)
    plt.ioff()

    print('Over!')



# 处理 NumPy 2.0 不兼容问题
def fix_numpy_compatibility():
    """修复 NumPy 2.0 兼容性问题"""
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")

    # 为旧代码提供向后兼容
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf
    if not hasattr(np, 'float128'):
        np.float128 = np.longdouble
    if not hasattr(np, 'float96'):
        np.float96 = np.longdouble

# 在导入其他库之前应用修复
fix_numpy_compatibility()



class V1_2_DQNInference_Visualizer:
    def __init__(self, modelPath, config):
        self.config = config
        self.modelPath = modelPath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化环境
        self.env = gym.make(self.config.environmentName, render_mode='rgb_array')
        self.numActions = self.env.action_space.n
        self.imageShape = (1, 120, 120)

        # 加载模型
        self.policyNetwork = self.loadModel()
        self.policyNetwork.eval()

        # 初始化matplotlib图形
        self.setupPlot()

        # 状态跟踪
        self.currentState = None
        self.totalReward = 0
        self.stepCount = 0
        self.episodeCount = 0

    def setupPlot(self):
        """设置matplotlib图形窗口"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('DQN Model Real-time Inference Visualization', fontsize=16, fontweight='bold')

        # 原始游戏画面
        self.ax1.set_title('Raw Game Screen')
        self.rawImage = self.ax1.imshow(np.zeros((210, 160, 3), dtype=np.uint8))
        self.ax1.axis('off')

        # 预处理后的灰度图
        self.ax2.set_title('Preprocessed Grayscale Frame')
        self.processedImage = self.ax2.imshow(np.zeros((120, 120), dtype=np.float32), cmap='gray')
        self.ax2.axis('off')

        # Q值分布
        self.ax3.set_title('Q-value Distribution for Each Action')
        self.qValuesBars = self.ax3.bar(range(self.numActions), np.zeros(self.numActions))
        self.ax3.set_xlabel('Action Index')
        self.ax3.set_ylabel('Q-value')
        self.ax3.set_xticks(range(self.numActions))

        # 统计信息
        self.ax4.set_title('Training Statistics')
        self.ax4.axis('off')
        self.statisticsText = self.ax4.text(0.1, 0.9, '', transform=self.ax4.transAxes, fontsize=12,
                                           verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

    def loadModel(self):
        """加载训练好的DQN模型 - 修复版本"""
        # 创建网络结构
        model = ResNetDeepQNetwork(self.imageShape, self.numActions).to(self.device)

        # 使用 weights_only=False 来加载包含自定义类的检查点
        checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=False)

        # 加载网络权重
        model.load_state_dict(checkpoint['policyNetworkState'])
        print("Model loaded successfully!")
        return model

    def preprocessFrame(self, frame):
        """预处理游戏帧（与训练时保持一致）"""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.imageShape[1], self.imageShape[2]), Image.BILINEAR)
        frame = np.array(img)
        frame = frame.astype(np.float32) / 255.0

        return frame

    def selectAction(self, state):
        """选择动作（贪婪策略）"""
        with torch.no_grad():
            # 确保状态张量格式正确
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if len(state.shape) == 2:  # 如果是2D，添加batch和channel维度
                state = state.unsqueeze(0).unsqueeze(0)

            state = state.to(self.device)
            qValues = self.policyNetwork(state)
            return qValues.max(1)[1].item(), qValues.cpu().numpy()[0]

    def resetEnvironment(self):
        """重置环境"""
        state, info = self.env.reset()
        processedState = self.preprocessFrame(state)
        self.currentState = processedState
        self.totalReward = 0
        self.stepCount = 0
        self.episodeCount += 1
        return state, processedState

    def stepEnvironment(self, action):
        """执行一步环境交互"""
        nextState, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        processedNextState = self.preprocessFrame(nextState)
        self.currentState = processedNextState
        self.totalReward += reward
        self.stepCount += 1
        return nextState, processedNextState, reward, done, info

    def updateVisualization(self, frame):
        """更新可视化显示"""
        # 更新原始游戏画面
        self.rawImage.set_array(frame)

        # 更新预处理后的灰度图
        self.processedImage.set_array(self.currentState)

        # 更新Q值分布
        _, qValues = self.selectAction(self.currentState)
        for bar, value in zip(self.qValuesBars, qValues):
            bar.set_height(value)

        maxQ = max(qValues) if len(qValues) > 0 else 1
        self.ax3.set_ylim(0, maxQ * 1.1 if maxQ > 0 else 1)

        # 更新统计信息
        statsText = f"""Episode: {self.episodeCount}
Step: {self.stepCount}
Total Reward: {self.totalReward:.2f}
Current Q-values:
"""
        for i, qVal in enumerate(qValues):
            statsText += f"  Action {i}: {qVal:.4f}\n"

        bestAction = np.argmax(qValues)
        statsText += f"Selected Action: {bestAction}"

        self.statisticsText.set_text(statsText)

        return self.rawImage, self.processedImage, *self.qValuesBars, self.statisticsText

    def runInference(self, maxSteps=1000):
        """运行推理并实时可视化"""
        print("Starting DQN inference visualization...")
        print("Close the visualization window to stop.")

        currentFrame, _ = self.resetEnvironment()
        done = False

        def update(frameNum):
            nonlocal currentFrame, done

            if done:
                currentFrame, _ = self.resetEnvironment()
                done = False

            # 选择动作
            action, _ = self.selectAction(self.currentState)

            # 执行动作
            currentFrame, processedFrame, reward, done, info = self.stepEnvironment(action)

            # 更新可视化
            artists = self.updateVisualization(currentFrame)

            # 如果游戏结束，打印统计信息
            if done:
                print(f"Episode {self.episodeCount} finished with total reward: {self.totalReward}")

            return artists

        # 创建动画
        self.animation = FuncAnimation(
            self.fig, update, frames=maxSteps,
            interval=100, blit=True, repeat=True, cache_frame_data=False
        )

        plt.show()

    def close(self):
        """关闭环境和资源"""
        self.env.close()
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        plt.close('all')




def PongDQNInference():
    # 调用DQN模型进行推理可视化    V1.2版本
    modelPath = '/home/next_lb/models/DQN_models/PongNoFrameskip-v4/best_model.pth'
    # 创建可视化器
    visualizer = V1_2_DQNInference_Visualizer(modelPath, TrainingConfig)

    # 运行推理可视化
    visualizer.runInference(maxSteps=10000)




def main():
    # CarRacingDQNInference()
    PongDQNInference()


if __name__ == '__main__':
    main()



