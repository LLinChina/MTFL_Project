"""
工具函数文件
包含数据加载、预处理、可视化等功能
"""

import os
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
                    # --- 关键修改开始 ---
                    # MTFL原始格式通常是: x1 x2 x3 x4 x5 y1 y2 y3 y4 y5
                    # 我们需要将其重组为: x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
                    raw_coords = [float(x) for x in parts[1:11]]
                    xs = raw_coords[0:5]
                    ys = raw_coords[5:10]
                    
                    landmarks = []
                    for x, y in zip(xs, ys):
                        landmarks.extend([x, y])
                    # --- 关键修改结束 ---
                    
                    landmarks = np.array(landmarks, dtype=np.float32)

                    # 性别标签 (通常在第12列，索引11)
                    # MTFL属性: gender, smile, glasses, head_pose
                    gender = int(parts[11]) 
                    # 假设原始标签: 1=Male, 2=Female (需根据实际数据集确认，这里沿用通用逻辑)
                    # 转换为: 0=Male, 1=Female
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

        orig_width, orig_height = image.size

        # 关键点归一化 [0, 1]
        landmarks = sample['landmarks'].copy()
        # 现在的landmarks已经是 x1, y1, x2, y2... 格式
        landmarks[0::2] = landmarks[0::2] / orig_width  # x
        landmarks[1::2] = landmarks[1::2] / orig_height # y

        if self.transform:
            image = self.transform(image)

        landmarks = torch.from_numpy(landmarks).float()
        gender = torch.tensor(sample['gender'], dtype=torch.long)

        return image, landmarks, gender


def get_transforms(is_train=True, img_size=224):
    if is_train:
        # 仅使用颜色增强，避免空间变换破坏关键点
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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