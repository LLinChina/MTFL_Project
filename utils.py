"""
工具函数文件
包含数据加载、预处理、可视化等功能
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MTFLDataset(Dataset):
    """
    MTFL数据集类
    """

    def __init__(self, data_root, annotation_file, transform=None, is_train=True):
        """
        Args:
            data_root: 数据根目录
            annotation_file: 标注文件路径 (training.txt 或 testing.txt)
            transform: 图像变换
            is_train: 是否为训练集
        """
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train

        # 解析标注文件
        self.samples = self._parse_annotation(annotation_file)

        print(f"Loaded {len(self.samples)} samples from {annotation_file}")

    def _parse_annotation(self, annotation_file):
        """解析标注文件"""
        samples = []

        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:  # 至少需要：图像路径 + 10个坐标 + 4个属性
                    continue

                # 提取图像路径
                img_path = parts[0]

                # 提取关键点坐标 (5个点，每个点x,y坐标)
                try:
                    landmarks = [float(x) for x in parts[1:11]]
                except ValueError:
                    continue

                # 提取性别标签 (第11个值)
                try:
                    gender = int(parts[11])  # 1:  Male, 2: Female
                    # 转换为0/1标签：0-Male, 1-Female
                    gender = 0 if gender == 1 else 1
                except (ValueError, IndexError):
                    continue

                samples.append({
                    'img_path': img_path,
                    'landmarks': np.array(landmarks, dtype=np.float32),
                    'gender': gender
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取图像
        img_path = os.path.join(self.data_root, sample['img_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}:  {e}")
            # 返回一个默认样本
            return self.__getitem__((idx + 1) % len(self))

        # 获取原始图像尺寸
        orig_width, orig_height = image.size

        # 关键点坐标归一化到[0, 1]
        landmarks = sample['landmarks'].copy()
        landmarks[0:: 2] = landmarks[0::2] / orig_width  # x坐标
        landmarks[1::2] = landmarks[1::2] / orig_height  # y坐标

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 转换为张量
        landmarks = torch.from_numpy(landmarks).float()
        gender = torch.tensor(sample['gender'], dtype=torch.long)

        return image, landmarks, gender


def get_transforms(is_train=True, img_size=224):
    """
    获取数据变换

    Args:
        is_train: 是否为训练集
        img_size: 图像尺寸

    Returns:
        transform: torchvision变换
    """
    if is_train:
        # 平衡策略：适度增强以提升泛化，但不破坏关键点坐标
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # 颜色增强 - 适度即可
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transform


def get_dataloaders(data_root, batch_size=32, num_workers=4, img_size=224):
    """
    获取数据加载器

    Args: 
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        img_size:  图像尺寸

    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 创建数据集
    train_dataset = MTFLDataset(
        data_root=data_root,
        annotation_file=os.path.join(data_root, 'training.txt'),
        transform=get_transforms(is_train=True, img_size=img_size),
        is_train=True
    )

    test_dataset = MTFLDataset(
        data_root=data_root,
        annotation_file=os.path.join(data_root, 'testing.txt'),
        transform=get_transforms(is_train=False, img_size=img_size),
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def calculate_nme(pred_landmarks, gt_landmarks, normalization='inter-ocular'):
    """
    计算归一化平均误差 (Normalized Mean Error)

    Args:
        pred_landmarks: 预测的关键点 (N, 10) 或 (N, 5, 2)
        gt_landmarks: 真实关键点 (N, 10) 或 (N, 5, 2)
        normalization: 归一化方式

    Returns:
        nme: 归一化平均误差
    """
    # 确保形状为 (N, 5, 2)
    if pred_landmarks.ndim == 2:
        pred_landmarks = pred_landmarks.reshape(-1, 5, 2)
    if gt_landmarks.ndim == 2:
        gt_landmarks = gt_landmarks.reshape(-1, 5, 2)

    # 计算欧氏距离
    distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=2))

    # 归一化因子（使用双眼间距）
    left_eye = gt_landmarks[:, 0, :]  # 左眼
    right_eye = gt_landmarks[:, 1, :]  # 右眼
    inter_ocular = np.sqrt(np.sum((left_eye - right_eye) ** 2, axis=1))

    # 计算NME
    nme = np.mean(distances / inter_ocular[:, np.newaxis])

    return nme


def visualize_predictions(image, pred_landmarks, pred_gender,
                          gt_landmarks=None, gt_gender=None,
                          save_path=None):
    """
    可视化预测结果

    Args: 
        image: 输入图像 (C, H, W) 张量
        pred_landmarks: 预测关键点 (10,)
        pred_gender: 预测性别 (0 or 1)
        gt_landmarks: 真实关键点（可选）
        gt_gender:  真实性别（可选）
        save_path: 保存路径
    """
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # 将归一化坐标转换回像素坐标
    h, w = img.shape[:2]
    pred_points = pred_landmarks.reshape(5, 2).copy()
    pred_points[:, 0] *= w
    pred_points[:, 1] *= h

    # 绘制预测关键点
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    labels = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']

    for i, (point, color, label) in enumerate(zip(pred_points, colors, labels)):
        ax.plot(point[0], point[1], 'o', color=color, markersize=10,
                label=f'Pred {label}')

    # 如果有真实标注，也绘制出来
    if gt_landmarks is not None:
        gt_points = gt_landmarks.reshape(5, 2).copy()
        gt_points[:, 0] *= w
        gt_points[:, 1] *= h

        for i, (point, color) in enumerate(zip(gt_points, colors)):
            ax.plot(point[0], point[1], 'x', color=color, markersize=10,
                    markeredgewidth=2)

    # 添加性别标签
    gender_text = "Female" if pred_gender == 1 else "Male"
    title = f"Predicted Gender: {gender_text}"

    if gt_gender is not None:
        gt_gender_text = "Female" if gt_gender == 1 else "Male"
        title += f" (GT: {gt_gender_text})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # 测试数据加载
    data_root = './data'  # 修改为实际路径

    if os.path.exists(data_root):
        train_loader, test_loader = get_dataloaders(
            data_root=data_root,
            batch_size=8,
            num_workers=0
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # 获取一个批次
        images, landmarks, gender = next(iter(train_loader))
        print(f"Image shape: {images.shape}")
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Gender shape: {gender.shape}")