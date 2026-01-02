"""
测试脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import get_model
from utils import (get_dataloaders, calculate_nme,
                   visualize_predictions, AverageMeter)

torch.serialization.add_safe_globals([np._core.multiarray.scalar])
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test Multi-Task Face Analysis Model')

    # 数据相关
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data root directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')

    # 模型相关
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_acc_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['base', 'improved'],
                        help='Model architecture type')

    # 测试相关
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 可视化相关
    parser.add_argument('--vis_num', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--vis_dir', type=str, default='./results',
                        help='Directory to save visualizations')

    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def test_model(model, test_loader, device, args):
    """测试模型并返回详细结果"""
    model.eval()

    # 初始化指标记录器
    gender_accs = AverageMeter()

    # 收集所有预测和真实值
    all_pred_landmarks = []
    all_gt_landmarks = []
    all_pred_gender = []
    all_gt_gender = []
    all_images = []

    print('Testing model...')
    with torch.no_grad():
        for images, landmarks, gender in tqdm(test_loader):
            images = images.to(device)
            landmarks = landmarks.to(device)
            gender = gender.to(device)

            batch_size = images.size(0)

            # 前向传播
            pred_landmarks, pred_gender = model(images)

            # 性别预测
            _, gender_pred = torch.max(pred_gender, 1)
            gender_acc = (gender_pred == gender).float().mean().item() * 100
            gender_accs.update(gender_acc, batch_size)

            # 收集结果
            all_pred_landmarks.append(pred_landmarks.cpu().numpy())
            all_gt_landmarks.append(landmarks.cpu().numpy())
            all_pred_gender.append(gender_pred.cpu().numpy())
            all_gt_gender.append(gender.cpu().numpy())
            all_images.append(images.cpu())

    # 合并所有批次
    all_pred_landmarks = np.concatenate(all_pred_landmarks, axis=0)
    all_gt_landmarks = np.concatenate(all_gt_landmarks, axis=0)
    all_pred_gender = np.concatenate(all_pred_gender, axis=0)
    all_gt_gender = np.concatenate(all_gt_gender, axis=0)
    all_images = torch.cat(all_images, dim=0)

    # 计算NME
    nme = calculate_nme(all_pred_landmarks, all_gt_landmarks)

    # 打印结果
    print('\n' + '=' * 50)
    print('Test Results:')
    print('=' * 50)
    print(f'Gender Classification Accuracy: {gender_accs.avg:.2f}%')
    print(f'Landmark Detection NME: {nme:.6f}')
    print('=' * 50 + '\n')

    return {
        'nme': nme,
        'accuracy': gender_accs.avg,
        'pred_landmarks': all_pred_landmarks,
        'gt_landmarks': all_gt_landmarks,
        'pred_gender': all_pred_gender,
        'gt_gender': all_gt_gender,
        'images': all_images
    }


def visualize_results(results, args):
    """可视化测试结果"""
    os.makedirs(args.vis_dir, exist_ok=True)

    num_samples = min(args.vis_num, len(results['images']))

    print(f'\nGenerating visualizations for {num_samples} samples...')

    # 随机选择一些样本
    indices = np.random.choice(len(results['images']), num_samples, replace=False)

    # 找出一些失败案例（性别分类错误或关键点误差大的）
    gender_errors = results['pred_gender'] != results['gt_gender']

    # 计算每个样本的NME
    sample_nmes = []
    for i in range(len(results['pred_landmarks'])):
        pred = results['pred_landmarks'][i: i + 1]
        gt = results['gt_landmarks'][i: i + 1]
        nme = calculate_nme(pred, gt)
        sample_nmes.append(nme)
    sample_nmes = np.array(sample_nmes)

    # 找出NME最大的样本（关键点检测失败案例）
    worst_nme_indices = np.argsort(sample_nmes)[-3:]

    # 找出性别分类错误的样本
    if np.any(gender_errors):
        gender_error_indices = np.where(gender_errors)[0][:2]
    else:
        gender_error_indices = []

    # 合并所有要可视化的索引
    vis_indices = np.unique(np.concatenate([
        indices[: 5],  # 随机样本
        worst_nme_indices,  # 最差关键点检测
        gender_error_indices  # 性别分类错误
    ]))

    for idx in vis_indices:
        image = results['images'][idx]
        pred_landmarks = results['pred_landmarks'][idx]
        pred_gender = results['pred_gender'][idx]
        gt_landmarks = results['gt_landmarks'][idx]
        gt_gender = results['gt_gender'][idx]

        # 判断是否为失败案例
        is_failure = (idx in worst_nme_indices or
                      idx in gender_error_indices)

        save_name = f'sample_{idx: 04d}'
        if is_failure:
            save_name += '_FAILURE'
        save_name += '.png'

        save_path = os.path.join(args.vis_dir, save_name)

        visualize_predictions(
            image=image,
            pred_landmarks=pred_landmarks,
            pred_gender=pred_gender,
            gt_landmarks=gt_landmarks,
            gt_gender=gt_gender,
            save_path=save_path
        )

    print(f'Visualizations saved to {args.vis_dir}')

    # 生成统计图表
    generate_statistics_plots(results, args)


def generate_statistics_plots(results, args):
    """生成统计图表"""

    # 1. 性别分类混淆矩阵
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(results['gt_gender'], results['pred_gender'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Male', 'Female'],
                yticklabels=['Male', 'Female'])
    plt.title('Gender Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.vis_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 2. NME分布直方图
    sample_nmes = []
    for i in range(len(results['pred_landmarks'])):
        pred = results['pred_landmarks'][i:i + 1]
        gt = results['gt_landmarks'][i:i + 1]
        nme = calculate_nme(pred, gt)
        sample_nmes.append(nme)

    plt.figure(figsize=(10, 6))
    plt.hist(sample_nmes, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(sample_nmes), color='red', linestyle='--',
                label=f'Mean NME: {np.mean(sample_nmes):.6f}')
    plt.xlabel('Normalized Mean Error (NME)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Landmark Detection Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.vis_dir, 'nme_distribution.png'), dpi=150)
    plt.close()

    print('Statistics plots saved')


def main():
    """主测试函数"""
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device:  {device}')

    # 加载数据
    print('Loading test data...')
    _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    # 创建模型
    print(f'Loading {args.model_type} model...')
    model = get_model(model_type=args.model_type, pretrained=False)

    # 加载训练好的权重
    # 加上 weights_only=False，告诉 PyTorch 信任这个文件
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f'Loaded model from epoch {checkpoint["epoch"]}')

    # 测试模型
    results = test_model(model, test_loader, device, args)

    # 可视化结果
    visualize_results(results, args)

    print('\nTesting completed!')


if __name__ == '__main__':
    main()