"""
训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import math

from model import get_model, WingLoss
from utils import get_dataloaders, calculate_nme, AverageMeter


# --- 不确定性加权多任务损失 (已激活) ---
class MultiTaskLoss(nn.Module):
    """基于不确定性的多任务损失自动加权"""
    def __init__(self):
        super().__init__()
        # 初始化 log_sigma
        # 性别任务简单但Loss小，给它更小的初始sigma(即更大的权重)
        self.log_sigma_landmark = nn.Parameter(torch.tensor(2.0)) # 初始权重较小
        self.log_sigma_gender = nn.Parameter(torch.tensor(-1.0))  # 初始权重较大
    
    def forward(self, loss_landmark, loss_gender):
        # 精度 precision = 1 / (2 * sigma^2)
        # Loss = precision * loss + log(sigma)
        precision_landmark = 0.5 * torch.exp(-self.log_sigma_landmark)
        precision_gender = 0.5 * torch.exp(-self.log_sigma_gender)
        
        total_loss = (precision_landmark * loss_landmark + 0.5 * self.log_sigma_landmark +
                      precision_gender * loss_gender + 0.5 * self.log_sigma_gender)
        return total_loss


# --- WingLoss ---
class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        # 降低放大倍数，防止Loss数值过大
        scale = 100.0 
        y_pred = pred * scale
        y_true = target * scale
        
        diff = torch.abs(y_pred - y_true)
        loss = torch.where(diff < self.w, 
                           self.w * torch.log(1 + diff / self.epsilon), 
                           diff - self.C)
        return torch.mean(loss)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Task Face Analysis Model')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-3) # 增加正则化
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    # 注意：启用自动权重后，命令行传入的 landmark_weight 将失效

    # 数据相关
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')

    # 模型相关
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'improved'],
                        help='Model architecture type')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')

    # 训练相关
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 损失权重 - 根据论文经验，landmark和gender基本平衡
    parser.add_argument('--landmark_weight', type=float, default=1.0,
                        help='Weight for landmark loss')
    parser.add_argument('--gender_weight', type=float, default=1.0,
                        help='Weight for gender loss')

    # 保存相关
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model, train_loader, criterion_landmark, criterion_gender,
                    optimizer, device, epoch, args, multi_task_loss_layer):
    """训练一个epoch"""
    model.train()
    # 确保权重层也在训练模式
    multi_task_loss_layer.train()

    losses = AverageMeter()
    landmark_losses = AverageMeter()
    gender_losses = AverageMeter()
    gender_accs = AverageMeter()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

    for images, landmarks, gender in pbar:
        images = images.to(device)
        landmarks = landmarks.to(device)
        gender = gender.to(device)
        batch_size = images.size(0)

        pred_landmarks, pred_gender = model(images)

        loss_landmark = criterion_landmark(pred_landmarks, landmarks)
        loss_gender = criterion_gender(pred_gender, gender)

        # --- 核心修改：使用自动加权层计算总损失 ---
        loss = multi_task_loss_layer(loss_landmark, loss_gender)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        _, gender_pred = torch.max(pred_gender, 1)
        gender_acc = (gender_pred == gender).float().mean().item() * 100

        losses.update(loss.item(), batch_size)
        landmark_losses.update(loss_landmark.item(), batch_size)
        gender_losses.update(loss_gender.item(), batch_size)
        gender_accs.update(gender_acc, batch_size)

        # 打印当前的自动权重 (1 / 2*exp(sigma))
        w_lm = 0.5 * torch.exp(-multi_task_loss_layer.log_sigma_landmark).item()
        w_gen = 0.5 * torch.exp(-multi_task_loss_layer.log_sigma_gender).item()

        pbar.set_postfix({
            'L_lm': f'{landmark_losses.avg:.2f}',
            'L_gen': f'{gender_losses.avg:.3f}',
            'Acc': f'{gender_accs.avg:.1f}%',
            'W_lm': f'{w_lm:.2f}', # 观察权重变化
            'W_gen': f'{w_gen:.2f}'
        })

    return losses.avg, landmark_losses.avg, gender_losses.avg, gender_accs.avg


def validate(model, test_loader, criterion_landmark, criterion_gender, device, args):
    model.eval()
    losses = AverageMeter()
    gender_accs = AverageMeter()
    all_pred_landmarks = []
    all_gt_landmarks = []

    with torch.no_grad():
        for images, landmarks, gender in tqdm(test_loader, desc='Validating'):
            images = images.to(device)
            landmarks = landmarks.to(device)
            gender = gender.to(device)
            batch_size = images.size(0)

            pred_landmarks, pred_gender = model(images)

            loss_landmark = criterion_landmark(pred_landmarks, landmarks)
            loss_gender = criterion_gender(pred_gender, gender)
            # 验证时简单求和即可，主要看指标
            loss = loss_landmark + loss_gender

            _, gender_pred = torch.max(pred_gender, 1)
            gender_acc = (gender_pred == gender).float().mean().item() * 100

            losses.update(loss.item(), batch_size)
            gender_accs.update(gender_acc, batch_size)
            all_pred_landmarks.append(pred_landmarks.cpu().numpy())
            all_gt_landmarks.append(landmarks.cpu().numpy())

    all_pred_landmarks = np.concatenate(all_pred_landmarks, axis=0)
    all_gt_landmarks = np.concatenate(all_gt_landmarks, axis=0)
    nme = calculate_nme(all_pred_landmarks, all_gt_landmarks)

    print(f'\nValidation Results:')
    print(f'  Gender Accuracy: {gender_accs.avg:.2f}%')
    print(f'  NME: {nme:.6f}\n')

    return losses.avg, nme, gender_accs.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading data...')
    train_loader, test_loader = get_dataloaders(args.data_root, args.batch_size, args.num_workers, args.img_size)

    print(f'Creating {args.model_type} model...')
    model = get_model(model_type=args.model_type, pretrained=args.pretrained).to(device)

    # --- 初始化自动权重层 ---
    multi_task_loss = MultiTaskLoss().to(device)

    criterion_landmark = WingLoss(w=10, epsilon=2)
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 定义损失函数
    # 使用Wing Loss for关键点检测 (CVPR 2018论文方法)
    criterion_landmark = WingLoss(omega=10, epsilon=2)
    # 性别分类使用标准交叉熵，添加标签平滑防止过拟合
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- 将自动权重层的参数加入优化器 ---
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': multi_task_loss.parameters(), 'lr': args.lr * 5} # 让权重参数学习得快一点
    ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    # 定义优化器 - 使用Adam with较小学习率
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度器 - 使用ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )

    # 训练循环
    best_nme = float('inf')
    best_acc = 0.0

    print('Starting training with Automatic Loss Balancing...')
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = train_one_epoch(
            model, train_loader, criterion_landmark, criterion_gender,
            optimizer, device, epoch, args, multi_task_loss # 传入权重层
        )

        val_loss, val_nme, val_acc = validate(
            model, test_loader, criterion_landmark, criterion_gender,
            device, args
        )

        scheduler.step(val_loss)
        # 更新学习率 - 基于验证loss
        scheduler.step(val_loss)
        
        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_nme_model.pth'))
            print(f'Saved best NME model (NME: {val_nme:.6f})')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_acc_model.pth'))
            print(f'Saved best Accuracy model (Acc: {val_acc:.2f}%)')

    print(f'Best NME: {best_nme:.6f}')
    print(f'Best Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()