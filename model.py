"""
模型定义文件
包含多任务学习网络结构
"""

import torch
import torch.nn as nn
from torchvision import models


class WingLoss(nn.Module):
    """
    Wing Loss for robust facial landmark detection
    论文: Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks (CVPR 2018)
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * torch.log(torch.tensor(1.0 + self.w / self.epsilon))

    def forward(self, pred, target):
        # 调整尺度，使Loss数值在合理范围
        scale = 100.0
        y_pred = pred * scale
        y_true = target * scale
        
        diff = torch.abs(y_pred - y_true)
        
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        return loss.mean()


class MultiTaskFaceNet(nn.Module):
    """
    基础版多任务网络 (Base)
    """
    def __init__(self, pretrained=True):
        super(MultiTaskFaceNet, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 10)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.landmark_head(features), self.gender_head(features)


class ImprovedMultiTaskFaceNet(nn.Module):
    """
    改进版多任务网络 (Improved)
    特点：更深的任务分支 (Deep Heads)，缓解任务冲突
    """
    def __init__(self, pretrained=True):
        super(ImprovedMultiTaskFaceNet, self).__init__()
        
        # 依然使用 ResNet18，但我们可以尝试解冻更多层或使用更强的预训练
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # --- 改进点：加深关键点分支 ---
        # 增加了一层 256 -> 128 的转换，让网络有更多参数去拟合几何变换
        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),      # 增加轻微Dropout防止过拟合
            nn.Linear(256, 128),  # 新增层
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 10)
        )

        # --- 改进点：加深性别分支 ---
        # 增加了一层，并保持较高的 Dropout，强迫网络学习鲁棒特征
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 新增层
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.landmark_head(features), self.gender_head(features)


def get_model(model_type='base', pretrained=True):
    """
    获取模型实例
    """
    if model_type == 'base':
        return MultiTaskFaceNet(pretrained=pretrained)
    elif model_type == 'improved':
        return ImprovedMultiTaskFaceNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # 测试模型
    model = get_model('base')
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(4, 3, 224, 224)
    landmarks, gender = model(dummy_input)
    print(f"Landmarks shape:  {landmarks.shape}")  # (4, 10)
    print(f"Gender shape:  {gender.shape}")  # (4, 2)