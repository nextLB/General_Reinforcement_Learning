

import torch
import torch.nn as nn
import torch.nn.functional as F


# 深度Q网络模型
class DQNNetwork(nn.Module):
    def __init__(self, inputShape, numActions, useBatchNorm):
        super(DQNNetwork, self).__init__()

        self.inputShape = inputShape
        self.numActions = numActions
        self.useBatchNorm = useBatchNorm

        # 改进的卷积层
        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32) if useBatchNorm else nn.Identity(),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64) if useBatchNorm else nn.Identity(),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128) if useBatchNorm else nn.Identity(),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256) if useBatchNorm else nn.Identity(),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化
        )

        # 计算卷积层输出维度
        self.convOutputSize = self._getConvOutputSize()

        # 全连接层
        self.fcLayers = nn.Sequential(
            nn.Linear(self.convOutputSize, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, numActions)
        )

        # 权重初始化
        self._initializeWeights()



    # 计算卷积层输出维度
    def _getConvOutputSize(self):
        with torch.no_grad():
            dummyInput = torch.zeros(1, *self.inputShape)
            convOutput = self.convLayers(dummyInput)
            return convOutput.view(1, -1).size(1)

    # 初始化网络权重
    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 获取特征维度
    def getFeatureSize(self):
        return self.convOutputSize


    # 保存模型
    def save(self, filePath):
        torch.save({
            'state_dict': self.state_dict(),
            'input_shape': self.inputShape,
            'num_actions': self.numActions,
            'use_batch_norm': self.useBatchNorm
        }, filePath)

    # 加载模型
    @classmethod
    def load(cls, filePath, device):
        checkpoint = torch.load(filePath, map_location=device)

        model = cls(
            checkpoint['input_shape'],
            checkpoint['num_actions'],
            checkpoint['use_batch_norm']
        )

        model.load_state_dict(checkpoint['state_dict'])
        if device:
            model = model.to(device)

        return model

    def forward(self, x):
        """前向传播"""
        # 确保输入张量维度正确
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # 调整维度顺序（如果需要）
        if x.shape[-1] == 3:  # HWC格式
            x = x.permute(0, 3, 1, 2)

        # 卷积层
        convFeatures = self.convLayers(x)

        # 展平特征
        flattenedFeatures = convFeatures.reshape(convFeatures.size(0), -1)

        # 全连接层
        qValues = self.fcLayers(flattenedFeatures)

        return qValues


