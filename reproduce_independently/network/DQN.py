
import torch.nn as nn
import torch
import torch.nn.functional as F


# ResNet基础残差块
class BasicBlock(nn.Module):
    """ResNet基础残差块"""

    def __init__(self, inChannels, outChannels, stride):
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            inChannels, outChannels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(outChannels)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            outChannels, outChannels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inChannels, outChannels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outChannels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class DQNNetWork(nn.Module):
    def __init__(self, imageShape, numActions):
        super(DQNNetWork, self).__init__()

        self.inputChannels = int(imageShape[2])
        self.inputHeight = int(imageShape[0])
        self.inputWidth = int(imageShape[1])

        # 初始卷积层
        self.conv1 = nn.Conv2d(
            self.inputChannels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1 = self._makeLayer(64, 64, blocks=2, stride=1)
        self.layer2 = self._makeLayer(64, 128, blocks=2, stride=2)
        self.layer3 = self._makeLayer(128, 256, blocks=2, stride=2)
        self.layer4 = self._makeLayer(256, 512, blocks=2, stride=2)

        # 自适应池化层
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

        # 计算全连接层输入维度
        fcInputSize = self._getFcInputSize()

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fcInputSize, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout防止过拟合
            nn.Linear(512, numActions)
        )

        # 初始化权重
        self._initializeWeights()


    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxPool(x)

        # ResNet块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 池化和展平
        x = self.avgPool(x)
        x = torch.flatten(x, 1)

        # 全连接层
        out = self.fc(x)

        return out


    def _makeLayer(self, inChannels, outChannels, blocks, stride):
        """创建ResNet层"""
        layers = []
        layers.append(BasicBlock(inChannels, outChannels, stride))

        for _ in range(1, blocks):
            layers.append(BasicBlock(outChannels, outChannels, stride=1))

        return nn.Sequential(*layers)



    def _getFcInputSize(self):
        """计算全连接层输入维度"""
        with torch.no_grad():
            dummyInput = torch.zeros(
                1, self.inputChannels, self.inputHeight, self.inputWidth
            )
            dummyOutput = self.conv1(dummyInput)
            dummyOutput = F.relu(self.bn1(dummyOutput))
            dummyOutput = self.maxPool(dummyOutput)

            dummyOutput = self.layer1(dummyOutput)
            dummyOutput = self.layer2(dummyOutput)
            dummyOutput = self.layer3(dummyOutput)
            dummyOutput = self.layer4(dummyOutput)

            dummyOutput = self.avgPool(dummyOutput)
            dummyOutput = torch.flatten(dummyOutput, 1)

            return dummyOutput.shape[1]



    def _initializeWeights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


