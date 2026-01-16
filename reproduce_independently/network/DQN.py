# # DQN.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DQNNetwork(nn.Module):
#     """深度Q网络模型"""
#
#     def __init__(self, inputShape, numActions, useBatchNorm=True):
#         super(DQNNetwork, self).__init__()
#
#         self.inputShape = inputShape
#         self.numActions = numActions
#         self.useBatchNorm = useBatchNorm
#
#         # 卷积层配置
#         self.convLayers = nn.Sequential(
#             # 第一层卷积
#             nn.Conv2d(inputShape[0], 32, kernel_size=8, stride=4),
#             nn.BatchNorm2d(32) if useBatchNorm else nn.Identity(),
#             nn.ReLU(inplace=True),
#
#             # 第二层卷积
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.BatchNorm2d(64) if useBatchNorm else nn.Identity(),
#             nn.ReLU(inplace=True),
#
#             # 第三层卷积
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.BatchNorm2d(64) if useBatchNorm else nn.Identity(),
#             nn.ReLU(inplace=True)
#         )
#
#         # 计算卷积层输出维度
#         self.convOutputSize = self._getConvOutputSize()
#
#         # 全连接层
#         self.fcLayers = nn.Sequential(
#             nn.Linear(self.convOutputSize, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#
#             nn.Linear(256, numActions)
#         )
#
#         # 权重初始化
#         self._initializeWeights()
#
#     def _getConvOutputSize(self):
#         """计算卷积层输出维度"""
#         with torch.no_grad():
#             dummyInput = torch.zeros(1, *self.inputShape)
#             convOutput = self.convLayers(dummyInput)
#             return convOutput.view(1, -1).size(1)
#
#     def _initializeWeights(self):
#         """初始化网络权重"""
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d):
#                 nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#             elif isinstance(module, nn.BatchNorm2d):
#                 nn.init.constant_(module.weight, 1)
#                 nn.init.constant_(module.bias, 0)
#             elif isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#
#     def forward(self, x):
#         """前向传播"""
#         # 确保输入张量维度正确
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#
#         # 调整维度顺序（如果需要）
#         if x.shape[-1] == 3:  # HWC格式
#             x = x.permute(0, 3, 1, 2)
#
#         # 卷积层
#         convFeatures = self.convLayers(x)
#
#         # 展平特征
#         flattenedFeatures = convFeatures.reshape(convFeatures.size(0), -1)
#
#         # 全连接层
#         qValues = self.fcLayers(flattenedFeatures)
#
#         return qValues
#
#     def getFeatureSize(self):
#         """获取特征维度"""
#         return self.convOutputSize
#
#     def save(self, filePath):
#         """保存模型"""
#         torch.save({
#             'state_dict': self.state_dict(),
#             'input_shape': self.inputShape,
#             'num_actions': self.numActions,
#             'use_batch_norm': self.useBatchNorm
#         }, filePath)
#
#     @classmethod
#     def load(cls, filePath, device=None):
#         """加载模型"""
#         checkpoint = torch.load(filePath, map_location=device)
#
#         model = cls(
#             checkpoint['input_shape'],
#             checkpoint['num_actions'],
#             checkpoint['use_batch_norm']
#         )
#
#         model.load_state_dict(checkpoint['state_dict'])
#         if device:
#             model = model.to(device)
#
#         return model
#
#
# class DuelingDQNNetwork(nn.Module):
#     """Dueling DQN网络（可选扩展）"""
#
#     def __init__(self, inputShape, numActions):
#         super(DuelingDQNNetwork, self).__init__()
#
#         # 共享特征提取层
#         self.featureLayers = nn.Sequential(
#             nn.Conv2d(inputShape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )
#
#         # 计算特征维度
#         with torch.no_grad():
#             dummyInput = torch.zeros(1, *inputShape)
#             featureOutput = self.featureLayers(dummyInput)
#             featureSize = featureOutput.view(1, -1).size(1)
#
#         # 价值流
#         self.valueStream = nn.Sequential(
#             nn.Linear(featureSize, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#
#         # 优势流
#         self.advantageStream = nn.Sequential(
#             nn.Linear(featureSize, 256),
#             nn.ReLU(),
#             nn.Linear(256, numActions)
#         )
#
#     def forward(self, x):
#         """前向传播"""
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#
#         if x.shape[-1] == 3:
#             x = x.permute(0, 3, 1, 2)
#
#         features = self.featureLayers(x)
#         features = features.reshape(features.size(0), -1)
#
#         value = self.valueStream(features)
#         advantage = self.advantageStream(features)
#
#         # 合并价值和优势
#         qValues = value + advantage - advantage.mean(dim=1, keepdim=True)
#
#         return qValues