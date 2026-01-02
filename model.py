"""
模型定义文件
包含多任务学习网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class WingLoss(nn.Module):
    """
    Wing Loss for robust facial landmark detection
    论文: Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks (CVPR 2018)
    """
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        
        # Wing loss分段函数
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        return loss.mean()


class MultiTaskFaceNet(nn.Module):
    """
    多任务人脸分析网络
    - 共享主干网络提取特征
    - 关键点检测分支（回归任务）
    - 性别分类分支（分类任务）
    """

    def __init__(self, pretrained=True):
        super(MultiTaskFaceNet, self).__init__()

        # 使用ResNet18作为主干网络
        resnet = models.resnet18(pretrained=pretrained)

        # 提取除最后全连接层外的所有层作为特征提取器
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 特征维度
        feature_dim = 512

        # 关键点检测分支 (5个点，每个点2个坐标，共10个输出)
        # 使用较深的网络，但dropout适中
        self.landmark_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10)  # 5个关键点 × 2个坐标
        )

        # 性别分类分支 (二分类：Male/Female)
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2个类别
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像张量 (batch_size, 3, 224, 224)

        Returns:
            landmarks: 关键点预测 (batch_size, 10)
            gender: 性别预测 logits (batch_size, 2)
        """
        # 特征提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # (batch_size, 512)

        # 关键点预测
        landmarks = self.landmark_head(features)

        # 性别预测
        gender = self.gender_head(features)

        return landmarks, gender


class ImprovedMultiTaskNet(nn.Module):
    """
    改进版多任务网络
    - 使用更深的ResNet50
    - 添加注意力机制
    - 任务特定的特征增强
    """

    def __init__(self, pretrained=True):
        super(ImprovedMultiTaskNet, self).__init__()

        # 使用ResNet50作为主干
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        feature_dim = 2048

        # 注意力模块
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid()
        )

        # 任务特定特征变换
        self.landmark_transform = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.gender_transform = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 关键点检测头
        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

        # 性别分类头
        self.gender_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 提取共享特征
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # 应用注意力
        attention_weights = self.attention(features)
        features = features * attention_weights

        # 任务特定特征
        landmark_features = self.landmark_transform(features)
        gender_features = self.gender_transform(features)

        # 任务预测
        landmarks = self.landmark_head(landmark_features)
        gender = self.gender_head(gender_features)

        return landmarks, gender


def get_model(model_type='base', pretrained=True):
    """
    获取模型实例

    Args: 
        model_type: 模型类型 ('base' 或 'improved')
        pretrained: 是否使用预训练权重

    Returns: 
        model: 模型实例
    """
    if model_type == 'base':
        return MultiTaskFaceNet(pretrained=pretrained)
    elif model_type == 'improved':
        return ImprovedMultiTaskNet(pretrained=pretrained)
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