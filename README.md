# MTFL_Project：多任务人脸关键点 + 性别分类

本项目使用 PyTorch 实现一个多任务网络：
- 任务1：5 点人脸关键点回归（输出 10 维，归一化到 [0,1]）
- 任务2：性别二分类（输出 2 类 logits）

工程主要脚本：
- 训练：[train.py](train.py)
- 测试与可视化：[test.py](test.py)
- 数据集与工具：[utils.py](utils.py)
- 模型定义：[model.py](model.py)

---

## 1. 依赖环境

### 1.1 Python / PyTorch
- Python：建议 **Python 3.9+**（你当前虚拟环境为 **Python 3.11.5**）
- PyTorch：要求 **torch>=2.0.0**（你当前环境为 **torch 2.7.0+cu128**）
- torchvision：要求 **torchvision>=0.15.0**（你当前环境为 **torchvision 0.22.0**）

### 1.2 其他依赖（可视化/评估/数据处理）
- matplotlib（你当前环境 3.10.0）
- Pillow（你当前环境 11.1.0）
- numpy（你当前环境 2.1.3）
- opencv-python（你当前环境 4.11.0.86）
- tqdm（你当前环境 4.67.1）
- scikit-learn（你当前环境 1.6.1）
- seaborn（你当前环境 0.13.2；用于 test.py 里的混淆矩阵可视化）

---

## 2. 安装与运行指令

### 2.1 安装

在项目根目录执行：

```bash
# 进入项目目录
cd D:\21309\DESKTOP\pychon\pycharm\MTFL_Project

# 建议创建虚拟环境（示例）
python -m venv .venv

# 激活虚拟环境（Windows PowerShell）
.venv\Scripts\Activate.ps1

# 安装依赖（推荐使用本项目提供的 requirements.txt）
pip install -r requirements.txt
```

#### 安装 GPU 版本 PyTorch（可选）
不同 CUDA 版本安装命令不同，请以 PyTorch 官网为准：https://pytorch.org/get-started/locally/

---

### 2.2 数据准备

将数据集放在项目根目录下的 `data/` 文件夹，至少需要：
- `data/training.txt`
- `data/testing.txt`
- 图片文件（可以在 `data/` 下按子目录组织，比如 `data/lfw_5590/`、`data/AFLW/`、`data/net_7876/` 等）

示例目录（与你当前工程结构一致）：

```
MTFL_Project/
  data/
    training.txt
    testing.txt
    AFLW/
    lfw_5590/
    net_7876/
```

---

### 2.3 训练

训练脚本为 [train.py](train.py)。关键参数：
- `--data_root`：数据根目录（默认 `./data`）
- `--model_type`：模型类型（`base` 或 `improved`）
- `--save_dir`：保存 checkpoint 的目录

#### 训练 base 模型（对应你历史的 No1~No6_base 等）

```bash
python train.py --data_root ./data --model_type base --save_dir ./checkpoints/No6_base --device cuda
```

#### 训练 improved 模型（你第七次实验 No7 使用 improved）

```bash
python train.py --data_root ./data --model_type improved --save_dir ./checkpoints/No7 --device cuda
```

训练过程会在 `--save_dir` 下输出：
- `best_acc_model.pth`：按性别准确率保存的最好模型
- `best_nme_model.pth`：按关键点 NME 保存的最好模型
- `checkpoint_epoch_XX.pth`：周期性保存的断点

你当前仓库中，各次实验的最优权重都已在各自目录下命名好（例如 [checkpoints/No7](checkpoints/No7) 里的 `best_acc_model.pth` / `best_nme_model.pth`）。

#### 断点续训（resume）

```bash
python train.py --data_root ./data --model_type improved --resume ./checkpoints/No7/checkpoint_epoch_50.pth --save_dir ./checkpoints/No7 --device cuda
```

---

### 2.4 测试（含可视化与统计图）

测试脚本为 [test.py](test.py)。它会：
- 计算性别分类准确率
- 计算关键点 NME
- 在 `--vis_dir` 输出若干可视化图片
- 输出混淆矩阵（需要 seaborn）和 NME 分布直方图

#### 测试 improved（No7）

```bash
python test.py --data_root ./data --model_type improved --model_path ./checkpoints/No7/best_acc_model.pth --device cuda
```

#### 测试 base（例如 No6_base）

```bash
python test.py --data_root ./data --model_type base --model_path ./checkpoints/No6_base/best_nme_model.pth --device cuda
```

你也可以把 `--model_path` 替换成任意一次实验目录下的 `best_acc_model.pth` 或 `best_nme_model.pth`。

---

## 3. 数据集存放路径说明

### 3.1 data_root
代码默认使用 `--data_root ./data`，因此：
- 标注文件应位于：
  - `./data/training.txt`
  - `./data/testing.txt`
- 图片路径应能通过 `os.path.join(data_root, img_path)` 找到。

### 3.2 标注文件格式
`training.txt` / `testing.txt` 每行格式：

```
<relative_image_path> x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 gender
```

说明：
- `relative_image_path`：相对 `data_root` 的路径
- `x1..x5, y1..y5`：5 个关键点的像素坐标（代码中会归一化到 [0,1]）
- `gender`：你的代码中做了转换：`gender = 0 if gender == 1 else 1`

---

## 4. requirements.txt

本项目提供标准依赖文件：[requirements.txt](requirements.txt)

安装命令：

```bash
pip install -r requirements.txt
```

注意：你当前仓库里还存在一个历史文件 `requirements. txt`（文件名中有空格/点），建议后续统一使用 `requirements.txt`。

---

## 5. Python 依赖包及版本

### 5.1 项目运行所需（最低版本约束）
以 [requirements.txt](requirements.txt) 为准。

### 5.2 当前虚拟环境的精确版本（可复现）
我已基于你当前虚拟环境生成精确版本清单：
- [requirements.lock.txt](requirements.lock.txt)

如果你希望完全复现当前环境，可使用：

```bash
pip install -r requirements.lock.txt
```

（提示：包含 CUDA 相关的 torch 版本时，可能需要你本机 CUDA/驱动匹配；如不匹配，请优先按 PyTorch 官网选择合适的 torch/torchvision 安装命令。）
