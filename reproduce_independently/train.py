
import json
import time
from datetime import datetime
from pathlib import Path
import torch
from agent.DQNAgent import DQNAgent



# 训练配置参数类
class Config:
    def __init__(self):
        # 版本信息
        self.version = 'V1'

        # 环境参数
        self.environmentName = "CarRacing-v3"
        self.renderMode = "rgb_array"
        self.lapCompletePercent = 0.95
        self.domainRandomize = False
        self.continuous = False

        # 训练参数
        self.totalEpisodes = 2000
        self.saveFrequency = 50
        self.validationFrequency = 10

        # DQN超参数
        self.gamma = 0.99
        self.epsilonStart = 1.0
        self.epsilonEnd = 0.01
        self.epsilonDecay = 0.995
        self.learningRate = 1e-4

        # 经验回放参数
        self.replayBufferCapacity = 30000
        self.batchSize = 8

        # 目标网络更新
        self.targetBufferCapacity = 100
        self.tau = 0.005    # 软更新参数

        # 设备配置（使用字符串表示，JSON可序列化）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 环境尺寸（将在初始化时设置）
        self.height = 0
        self.width = 0
        self.channels = 0

        # 路径配置
        self.baseModelDir = "models"



    # 获取torch.device对象
    def getDevice(self):
        return torch.device(self.device)
    # 将配置转换为字典（JSON可序列化）
    def toDict(self):
        configDict = {}
        for key, value in self.__dict__.items():
            # 跳过私有属性
            if key.startswith('_'):
                continue


            # 处理特殊类型
            if key == 'device':
                configDict[key] = str(value)    # 确保device是字符串
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                configDict[key] = value
            else:
                # 将其他类型转换为字符串
                configDict[key] = str(value)

        return configDict
    # 保存配置到文件
    def save(self, filePath):
        with open(filePath, 'w') as f:
            json.dump(self.toDict(), f, indent=4, ensure_ascii=False)
    # 从文件加载配置
    @classmethod
    def load(cls, filePath):
        config = cls()
        with open(filePath, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                # 特殊处理device字段
                if hasattr(config, key):
                    config.device = value
                else:
                    setattr(config,key, value)

        return config



# 训练日志记录器
class TrainingLogger:
    def __init__(self, logDir):
        self.logDir = Path(logDir)
        self.logDir.mkdir(parents=True, exist_ok=True)

        self.rewards = []
        self.losses = []
        self.episodeTimes = []
        self.epsilonValues = []

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logFile = self.logDir / f"training_log_{timestamp}.txt"
        self.metricsFile = self.logDir / f"metrics_{timestamp}.json"
    # 记录单回合训练结果
    def logEpisode(self, episode, totalReward, loss, epsilon, episodeTime):
        self.rewards.append(float(totalReward))
        if loss is not None:
            self.losses.append(float(loss))
        self.epsilonValues.append(float(epsilon))
        self.episodeTimes.append(float(episodeTime))

        # 修复f-string中的三元表达式问题
        lossDisplay = f"{loss:.4f}" if loss is not None else "0.0000"

        logMessage = (
            f"Episode {episode}: "
            f"Reward={totalReward:.2f}, "
            f"Loss={lossDisplay}, "
            f"Epsilon={epsilon:.4f}, "
            f"Time={episodeTime:.2f}s"
        )

        self._writeToLog(logMessage)
    # 记录单步训练信息
    def logStep(self, step, reward, bufferSize, epsilon):
        logMessage = (
            f"Step {step}: "
            f"Reward={reward:.2f}, "
            f"Buffer={bufferSize}, "
            f"Epsilon={epsilon:.3f}"
        )

        self._writeToLog(logMessage)
    # 写入日志文件
    def _writeToLog(self, message):
        with open(self.logFile, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    # 保存训练指标(JSON可序列化)
    def saveMetrics(self):
        metrics = {
            'rewards': self.rewards,
            'losses': self.losses,
            'epsilonValues': self.epsilonValues,
            'episodeTimes': self.episodeTimes,
            'totalEpisodes': len(self.rewards),
            'averageReward': float(sum(self.rewards) / len(self.rewards)) if self.rewards else 0,
            'maxReward': float(max(self.rewards)) if self.rewards else 0,
            'minReward': float(min(self.rewards)) if self.rewards else 0
        }

        with open(self.metricsFile, 'w') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)





# 训练器主类
class Trainer:

    def __init__(self, config):
        self.config = config
        self.agent = None
        self.logger = None


    # 设置训练环境
    def setupTraining(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainingName = f"car_racing_{self.config.version}_{timestamp}"
        self.saveDir = Path(self.config.baseModelDir) / trainingName
        self.saveDir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        configPath = self.saveDir / "config.json"
        self.config.save(configPath)

        # 初始化日志记录器
        self.logger = TrainingLogger(self.saveDir)

        # 初始化智能体
        self.agent = DQNAgent(self.config)

        # 打印训练信息
        self._printTrainingInfo()




    # 执行训练
    def train(self):
        print("开始训练...")
        startTime = time.time()

        try:
            for episode in range(1, self.config.totalEpisodes + 1):
                episodeStartTime = time.time()

                # 训练一个回合
                totalReward, averageLoss = self.agent.trainEpisode()

                episodeTime = time.time() - episodeStartTime

                # 记录训练结果
                self.logger.logEpisode(
                    episode, totalReward, averageLoss,
                    self.agent.epsilon, episodeTime
                )

                # 定期保存模型
                if episode % self.config.saveFrequency == 0:
                    self._saveModel(episode)

                # 定期验证
                if episode % self.config.validationFrequency == 0:
                    self._validateModel(episode)

            # 训练完成
            totalTime = time.time() - startTime
            self._completeTraining(totalTime)


        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self._saveInterruptedTraining(startTime)
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")






    # 打印训练信息
    def _printTrainingInfo(self):
        print("=" * 60)
        print(f"训练配置:")
        print(f"  环境: {self.config.environmentName}")
        print(f"  设备: {self.config.device}")
        print(f"  总回合数: {self.config.totalEpisodes}")
        print(f"  保存目录: {self.saveDir}")
        print(f"  动作空间大小: {self.agent.actionSpace}")
        print(f"  输入形状: ({self.config.channels}, {self.config.height}, {self.config.width})")
        print("=" * 60)
    # 保存模型
    def _saveModel(self, episode):
        modelPath = self.saveDir / f"model_episode_{episode}.pth"
        self.agent.saveModel(str(modelPath))
        print(f"模型已保存: {modelPath}")
    # 验证模型性能
    def _validateModel(self, episode):
        try:
            validationReward = self.agent.validate()
            print(f"验证回合 {episode}: 奖励 = {validationReward:.2f}")
        except Exception as e:
            print(f"验证回合 {episode} 失败: {e}")
    # 完成训练
    def _completeTraining(self, totalTime):
        """完成训练"""
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"总耗时: {totalTime:.2f}秒")
        print(f"平均每回合耗时: {totalTime / self.config.totalEpisodes:.2f}秒")

        # 保存最终模型
        finalModelPath = self.saveDir / "model_final.pth"
        self.agent.saveModel(str(finalModelPath))
        print(f"最终模型已保存: {finalModelPath}")

        # 保存训练指标
        self.logger.saveMetrics()
        print(f"训练指标已保存: {self.logger.metricsFile}")

        # 生成训练摘要
        self._generateTrainingSummary()
    # 保存被中断的训练
    def _saveInterruptedTraining(self, startTime):
        totalTime = time.time() - startTime
        print(f"\n训练已运行 {totalTime:.2f}秒")

        if self.agent and self.agent.episodeCount > 0:
            # 保存当前模型
            interruptedModelPath = self.saveDir / "model_interrupted.pth"
            self.agent.saveModel(str(interruptedModelPath))
            print(f"中断时模型已保存: {interruptedModelPath}")

            # 保存当前指标
            self.logger.saveMetrics()

            # 生成中断摘要
            self._generateInterruptedSummary(totalTime)
    # 生成训练摘要
    def _generateTrainingSummary(self):
        summaryPath = self.saveDir / "training_summary.txt"
        with open(summaryPath, 'w') as f:
            f.write("训练摘要\n")
            f.write("=" * 60 + "\n")
            f.write(f"训练时间: {datetime.now()}\n")
            f.write(f"总回合数: {self.config.totalEpisodes}\n")
            f.write(f"最终探索率: {self.agent.epsilon:.4f}\n")
            f.write(f"经验缓冲区大小: {self.agent.getBufferSize()}\n")
            f.write(f"总训练步数: {self.agent.stepCount}\n")
            if self.logger.rewards:
                f.write(f"平均奖励: {sum(self.logger.rewards) / len(self.logger.rewards):.2f}\n")
                f.write(f"最大奖励: {max(self.logger.rewards):.2f}\n")
                f.write(f"最小奖励: {min(self.logger.rewards):.2f}\n")
    # 生成中断训练摘要
    def _generateInterruptedSummary(self, totalTime):
        summaryPath = self.saveDir / "training_interrupted_summary.txt"
        with open(summaryPath, 'w') as f:
            f.write("训练中断摘要\n")
            f.write("=" * 60 + "\n")
            f.write(f"中断时间: {datetime.now()}\n")
            f.write(f"已完成回合数: {self.agent.episodeCount}\n")
            f.write(f"训练时长: {totalTime:.2f}秒\n")
            f.write(f"最终探索率: {self.agent.epsilon:.4f}\n")
            f.write(f"经验缓冲区大小: {self.agent.getBufferSize()}\n")
            f.write(f"总训练步数: {self.agent.stepCount}\n")
            if self.logger.rewards:
                f.write(f"平均奖励: {sum(self.logger.rewards) / len(self.logger.rewards):.2f}\n")






# 主函数
def main():
    try:
        config = Config()
        trainer = Trainer(config)
        trainer.setupTraining()
        trainer.train()


    except Exception as e:
        print(f"训练初始化失败: {e}")
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    main()

