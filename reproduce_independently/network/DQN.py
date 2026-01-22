
import torch.nn as nn
import torch





class DQNNetWork(nn.Module):
    def __init__(self, imageShape, numActions):
        super(DQNNetWork, self).__init__()

        self.inputChannels = int(imageShape[2])
        self.inputHeight = int(imageShape[0])
        self.inputWidth = int(imageShape[1])

        # 卷积层提取特征
        self.convLayers = nn.Sequential(
            nn.Conv2d(self.inputChannels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算卷积层输出维度
        convOutputSize = self._get_conv_output_size()

        # 全连接层输出Q值
        self.fc = nn.Sequential(
            nn.Linear(convOutputSize, 512),
            nn.ReLU(),
            nn.Linear(512, numActions)
        )

    def forward(self, x):
        x = self.convLayers(x)
        out = self.fc(x)
        return out


    def _get_conv_output_size(self):
        """计算卷积层输出维度"""
        # 创建一个假数据来获取卷积层输出大小
        with torch.no_grad():
            dummyInput = torch.zeros(1, self.inputChannels, self.inputHeight, self.inputWidth)
            dummyOutput = self.convLayers(dummyInput)
            return dummyOutput.shape[1]




