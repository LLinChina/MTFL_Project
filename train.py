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

from model import get_model
from utils import get_dataloaders, calculate_nme, AverageMeter

# --- 新增 WingLoss ---
class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        # 将归一化坐标放大，使WingLoss在合适的尺度工作
        y_pred = pred * 224.0
        y_true = target * 224.0
        
        diff = torch.abs(y_pred - y_true)
        loss = torch.where(diff < self.w, 
                           self.w * torch.log(1 + diff / self.epsilon), 
                           diff - self.C)
        return torch.mean(loss)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Multi-Task Face Analysis Model')

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
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 损失权重 - 增加性别分类任务的权重
    parser.add_argument('--landmark_weight', type=float, default=0.8,
                        help='Weight for landmark loss')
    parser.add_argument('--gender_weight', type=float, default=1.5,
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
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model, train_loader, criterion_landmark, criterion_gender,
                    optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()

    # 创建指标记录器
    losses = AverageMeter()
    landmark_losses = AverageMeter()
    gender_losses = AverageMeter()
    gender_accs = AverageMeter()

    # 进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

    for images, landmarks, gender in pbar:
        # 数据移至设备
        images = images.to(device)
        landmarks = landmarks.to(device)
        gender = gender.to(device)

        batch_size = images.size(0)

        # 前向传播
        pred_landmarks, pred_gender = model(images)

        # 计算损失
        loss_landmark = criterion_landmark(pred_landmarks, landmarks)
        loss_gender = criterion_gender(pred_gender, gender)

        # 总损失（加权和）
        loss = (args.landmark_weight * loss_landmark +
                args.gender_weight * loss_gender)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 计算性别分类准确率
        _, gender_pred = torch.max(pred_gender, 1)
        gender_acc = (gender_pred == gender).float().mean().item() * 100

        # 更新指标
        losses.update(loss.item(), batch_size)
        landmark_losses.update(loss_landmark.item(), batch_size)
        gender_losses.update(loss_gender.item(), batch_size)
        gender_accs.update(gender_acc, batch_size)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lm_loss': f'{landmark_losses.avg:.4f}',
            'gen_loss': f'{gender_losses.avg:.4f}',
            'gen_acc': f'{gender_accs.avg:.2f}%'
        })

    return losses.avg, landmark_losses.avg, gender_losses.avg, gender_accs.avg


def validate(model, test_loader, criterion_landmark, criterion_gender, device, args):
    """验证模型"""
    model.eval()

    losses = AverageMeter()
    landmark_losses = AverageMeter()
    gender_losses = AverageMeter()
    gender_accs = AverageMeter()

    # 用于计算NME
    all_pred_landmarks = []
    all_gt_landmarks = []

    with torch.no_grad():
        for images, landmarks, gender in tqdm(test_loader, desc='Validating'):
            images = images.to(device)
            landmarks = landmarks.to(device)
            gender = gender.to(device)

            batch_size = images.size(0)

            # 前向传播
            pred_landmarks, pred_gender = model(images)

            # 计算损失
            loss_landmark = criterion_landmark(pred_landmarks, landmarks)
            loss_gender = criterion_gender(pred_gender, gender)
            loss = (args.landmark_weight * loss_landmark +
                    args.gender_weight * loss_gender)

            # 性别准确率
            _, gender_pred = torch.max(pred_gender, 1)
            gender_acc = (gender_pred == gender).float().mean().item() * 100

            # 更新指标
            losses.update(loss.item(), batch_size)
            landmark_losses.update(loss_landmark.item(), batch_size)
            gender_losses.update(loss_gender.item(), batch_size)
            gender_accs.update(gender_acc, batch_size)

            # 收集关键点用于计算NME
            all_pred_landmarks.append(pred_landmarks.cpu().numpy())
            all_gt_landmarks.append(landmarks.cpu().numpy())

    # 计算NME
    all_pred_landmarks = np.concatenate(all_pred_landmarks, axis=0)
    all_gt_landmarks = np.concatenate(all_gt_landmarks, axis=0)
    nme = calculate_nme(all_pred_landmarks, all_gt_landmarks)

    print(f'\nValidation Results:')
    print(f'  Loss: {losses.avg:.4f}')
    print(f'  Landmark Loss: {landmark_losses.avg:.4f}')
    print(f'  Gender Loss: {gender_losses.avg:.4f}')
    print(f'  Gender Accuracy: {gender_accs.avg:.2f}%')
    print(f'  NME: {nme:.6f}\n')

    return losses.avg, nme, gender_accs.avg


def main():
    """主训练函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    print('Loading data...')
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    # 创建模型
    print(f'Creating {args.model_type} model.. .')
    model = get_model(model_type=args.model_type, pretrained=args.pretrained)
    model = model.to(device)

    # 定义损失函数
    criterion_landmark = WingLoss(w=10, epsilon=2)  # 使用WingLoss代替Smooth L1
    criterion_gender = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑

    # 定义优化器 - 使用AdamW和更优的参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度器：ReduceLROnPlateau微调
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # 训练循环
    best_nme = float('inf')
    best_acc = 0.0

    print('Starting training...')
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_lm_loss, train_gen_loss, train_gen_acc = train_one_epoch(
            model, train_loader, criterion_landmark, criterion_gender,
            optimizer, device, epoch, args
        )

        # 验证
        val_loss, val_nme, val_acc = validate(
            model, test_loader, criterion_landmark, criterion_gender,
            device, args
        )

        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')

        # 保存最佳模型
        if val_nme < best_nme:
            best_nme = val_nme
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nme': val_nme,
                'acc': val_acc,
            }, os.path.join(args.save_dir, 'best_nme_model.pth'))
            print(f'Saved best NME model (NME: {val_nme:.6f})')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nme': val_nme,
                'acc': val_acc,
            }, os.path.join(args.save_dir, 'best_acc_model.pth'))
            print(f'Saved best Accuracy model (Acc: {val_acc:.2f}%)')

        # 定期保存检查点
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nme': val_nme,
                'acc': val_acc,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    print('Training completed!')
    print(f'Best NME: {best_nme:. 6f}')
    print(f'Best Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()