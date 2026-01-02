"""
工具函数文件
包含数据加载、预处理、可视化等功能
"""

import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class MTFLDataset(Dataset):
    """
    MTFL数据集类 - 增强版 (包含几何变换坐标同步)
    """

    def __init__(self, data_root, annotation_file, transform=None, is_train=True):
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train
        self.samples = self._parse_annotation(annotation_file)
        print(f"Loaded {len(self.samples)} samples from {annotation_file}")

    def _parse_annotation(self, annotation_file):
        """解析标注文件"""
        samples = []
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            return []

        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 12: 
                    continue

                img_path = parts[0]

                try:
                    # 解析坐标 x1, x2... y1, y2...
                    raw_coords = [float(x) for x in parts[1:11]]
                    xs = raw_coords[0:5]
                    ys = raw_coords[5:10]
                    
                    # 重组为 x1, y1, x2, y2...
                    landmarks = []
                    for x, y in zip(xs, ys):
                        landmarks.extend([x, y])
                    
                    landmarks = np.array(landmarks, dtype=np.float32)

                    # 性别标签处理
                    gender = int(parts[11]) 
                    gender = 0 if gender == 1 else 1
                    
                except ValueError:
                    continue

                samples.append({
                    'img_path': img_path,
                    'landmarks': landmarks,
                    'gender': gender
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample['img_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

        w, h = image.size
        landmarks = sample['landmarks'].copy() # 原始像素坐标

        # --- 核心：带坐标更新的几何增强 (仅训练时启用) ---
        if self.is_train:
            # 1. 随机旋转 (-15 ~ 15度)
            if random.random() < 0.5: # 50%概率旋转
                angle = random.uniform(-15, 15)
                image = F.rotate(image, angle)
                
                # 坐标旋转数学公式
                # 旋转中心是图片中心
                cx, cy = w / 2, h / 2
                # 角度转弧度 (注意：图像坐标系y轴向下，逆时针旋转需取反)
                rad = math.radians(-angle) 
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                new_lms = []
                for i in range(0, 10, 2):
                    px, py = landmarks[i], landmarks[i+1]
                    # 平移到中心 -> 旋转 -> 平移回原位
                    nx = (px - cx) * cos_a - (py - cy) * sin_a + cx
                    ny = (px - cx) * sin_a + (py - cy) * cos_a + cy
                    new_lms.extend([nx, ny])
                landmarks = np.array(new_lms)

            # 2. 随机缩放裁切 (Scale 0.85 ~ 1.0)
            # 这模拟了人脸在图片中忽大忽小的情况，强迫模型学习尺度不变性
            if random.random() < 0.5:
                scale = random.uniform(0.85, 1.0)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # 随机选择左上角裁剪点
                if w > new_w and h > new_h:
                    dx = random.randint(0, w - new_w)
                    dy = random.randint(0, h - new_h)
                    
                    image = F.crop(image, dy, dx, new_h, new_w)
                    
                    # 坐标更新：所有点减去偏移量
                    landmarks[0::2] -= dx
                    landmarks[1::2] -= dy
                    
                    # 更新当前宽高，用于后续归一化
                    w, h = new_w, new_h

        # --- 增强结束 ---

        # 归一化 [0, 1]
        landmarks[0::2] /= w
        landmarks[1::2] /= h
        
        # 边界截断：防止增强后关键点跑出图片范围 (0~1之外)
        landmarks = np.clip(landmarks, 0.0, 1.0)

        if self.transform:
            image = self.transform(image)

        landmarks = torch.from_numpy(landmarks).float()
        gender = torch.tensor(sample['gender'], dtype=torch.long)

        return image, landmarks, gender


def get_transforms(is_train=True, img_size=224):
    # 几何变换已经在 __getitem__ 里手动处理了
    # 这里只做 Resize 和 颜色变换
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # 颜色抖动：改变亮度、对比度、饱和度，防止模型死记硬背肤色
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def get_dataloaders(data_root, batch_size=32, num_workers=4, img_size=224):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def calculate_nme(pred_landmarks, gt_landmarks):
    """计算NME"""
    if pred_landmarks.ndim == 2:
        pred_landmarks = pred_landmarks.reshape(-1, 5, 2)
    if gt_landmarks.ndim == 2:
        gt_landmarks = gt_landmarks.reshape(-1, 5, 2)

    # 欧氏距离
    distances = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=2))
    
    # 双眼间距归一化
    left_eye = gt_landmarks[:, 0, :]
    right_eye = gt_landmarks[:, 1, :]
    inter_ocular = np.sqrt(np.sum((left_eye - right_eye) ** 2, axis=1)) + 1e-6

    nme = np.mean(np.mean(distances, axis=1) / inter_ocular)
    return nme


def visualize_predictions(image, pred_landmarks, pred_gender, gt_landmarks=None, gt_gender=None, save_path=None):
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    h, w = img.shape[:2]

    # 预测点 (红色)
    pred_pts = pred_landmarks.reshape(5, 2)
    plt.scatter(pred_pts[:, 0] * w, pred_pts[:, 1] * h, c='r', s=50, label='Pred')

    # 真实点 (绿色)
    if gt_landmarks is not None:
        gt_pts = gt_landmarks.reshape(5, 2)
        plt.scatter(gt_pts[:, 0] * w, gt_pts[:, 1] * h, c='g', s=30, marker='x', label='GT')

    title = f"Pred: {'Female' if pred_gender==1 else 'Male'}"
    if gt_gender is not None:
        title += f" | GT: {'Female' if gt_gender==1 else 'Male'}"
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

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