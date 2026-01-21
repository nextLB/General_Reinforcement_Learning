
import torch.nn as nn






class DQNNetWork(nn.Module):
    def __init__(self, imageShape):
        super(DQNNetWork, self).__init__()

        # 卷积层提取特征
        self.convLayers = nn.Sequential(
            nn.Conv2d(imageShape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.convLayers(x)



