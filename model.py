"""
模型定义文件
包含多任务学习网络结构
"""

import torch
import torch.nn as nn
from torchvision import models


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

        # 关键点检测分支 (5个点，每个点2个坐标，共10个输出)
        self.landmark_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 10)  # 输出10个坐标
        )

        # 性别分类分支 (二分类：Male/Female)
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
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