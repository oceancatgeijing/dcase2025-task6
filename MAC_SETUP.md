# Mac 上运行 DCASE2025 Task 6 Baseline 指南

本指南将帮助您在 Mac 上设置和运行 DCASE2025 Task 6 的基线系统。

## 快速开始 (Apple Silicon Mac)

如果您使用的是 Apple Silicon (M1/M2/M3) Mac，可以快速开始：

```bash


# 1. 进入项目目录
cd /Users/seki/Desktop/毕业设计/dcase2025_task6_baseline

# 2. 创建并激活环境
conda create -n d25_t6 python=Python 3.10.19 -y
conda activate d25_t6

# 3. 安装依赖
brew install p7zip  # 或 conda install -c conda-forge p7zip
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt

# 4. 测试环境配置
python test_mac_env.py

# 5. 测试设备 (可选)
python mac_device_helper.py

# 6. 开始训练 (仅下载数据)
python -m d25_t6.train --no-train --no-test --data_path=data

# 7. 开始训练
python -m d25_t6.train \
  --data_path=data \
  --batch_size=8 \
  --batch_size_eval=8 \
  --n_workers=4 \
  --devices=1 \
  --no-compile \
  --seed=13

#cpu模式
python -m d25_t6.train \
  --data_path=data \
  --batch_size=4 \  
  --batch_size_eval=4 \
  --n_workers=4 \
  --devices=0 \  
  --no-compile \
  --seed=13
```

## 系统要求

- macOS (建议 macOS 12.0 或更高版本)
- Python 3.11
- 至少 8GB RAM (推荐 16GB 或更多)
- 足够的磁盘空间用于数据集 (至少 200GB)

## 步骤 1: 安装 Homebrew (如果尚未安装)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 步骤 2: 安装 Conda

### 选项 A: 使用 Miniconda (推荐)

```bash
# 下载 Miniconda for macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# 如果是 Apple Silicon (M1/M2/M3)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 安装
bash Miniconda3-latest-MacOSX-x86_64.sh  # 或 arm64.sh
```

### 选项 B: 使用 Homebrew

```bash
brew install miniconda
```

## 步骤 3: 创建 Conda 环境

```bash
# 进入项目目录
cd /Users/seki/Desktop/毕业设计/dcase2025_task6_baseline

# 创建 Python 3.11 环境
conda create -n d25_t6 python=3.11 -y

# 激活环境
conda activate d25_t6
```

## 步骤 4: 安装 7z

```bash
# 使用 Homebrew 安装
brew install p7zip

# 或者使用 conda
conda install -c conda-forge p7zip -y
```

## 步骤 5: 安装 PyTorch

### 对于 Apple Silicon (M1/M2/M3) Mac

PyTorch 支持 MPS (Metal Performance Shaders) 加速：

```bash
# 安装支持 MPS 的 PyTorch
pip3 install torch torchvision torchaudio
```

### 对于 Intel Mac

```bash
# 安装 CPU 版本的 PyTorch
pip3 install torch torchvision torchaudio
```

验证安装：

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}' if hasattr(torch.backends, 'mps') else 'MPS not available')"
```

## 步骤 6: 安装其他依赖

```bash
# 确保在项目根目录
cd /Users/seki/Desktop/毕业设计/dcase2025_task6_baseline

# 安装依赖
pip3 install -r requirements.txt
```

## 步骤 7: 配置 Weights and Biases (可选但推荐)

```bash
# 登录 wandb (首次使用需要)
wandb login

# 如果不想使用 wandb，可以在运行时添加 --no-logging 参数
```

## 步骤 8: 运行训练

### 基本训练 (仅 ClothoV2 数据集)

**对于 Apple Silicon Mac (M1/M2/M3):**

```bash
python -m d25_t6.train \
  --data_path=data \
  --batch_size=8 \
  --batch_size_eval=8 \
  --n_workers=4 \
  --devices=1 \
  --no-compile \
  --seed=13

python -m d25_t6.train 
	--data_path=data 
	--batch_size=2 
	--batch_size_eval=2 
	--n_workers=1 
	--no-compile 
	--seed=13

# transformer 训练方式
python -m d25_t6.train \
    --data_path=data \
    --batch_size=8 \
    --batch_size_eval=8 \
    --n_workers=8 \
    --no-compile \
    --seed=13 \
    --training_mode transformer

# bi-encoder 训练方式
python -m d25_t6.train \
    --data_path=data \
    --batch_size=8 \
    --batch_size_eval=8 \
    --n_workers=4 \
    --no-compile \
    --seed=13 \
    --training_mode bi-encoder

# 设置学术环境
source /etc/network_turbo
# 5090上跑transformer
python -m d25_t6.train     --data_path=data     --batch_size=32     --batch_size_eval=32     --n_workers=8     --s_patchout_t=15     --s_patchout_f=2     --no-compile     --training_mode transformer 

# 跑biencoder模式（bi-encoder）
python -m d25_t6.train     --data_path=data     --batch_size=32     --batch_size_eval=32     --n_workers=8     --s_patchout_t=15     --s_patchout_f=2     --no-compile     --training_mode bi-encoder 


#win 不设置wandb
python -m d25_t6.train --data_path=data --batch_size=8 --batch_size_eval=8 --n_workers=8 --no-compile --seed=13 --training_mode bi-encoder --no-logging

python -m d25_t6.train 
--training_mode transformer 
--devices 1 
--n_workers 4 
--batch_size 2
```

# git语句
git status
git add .
git commit -m "内容"
git push origin main


**对于 Intel Mac:**

```bash
python -m d25_t6.train \
  --data_path=data \
  --batch_size=4 \
  --batch_size_eval=4 \
  --n_workers=2 \
  --devices=1 \
  --no-compile \
  --seed=13
```

### 包含 AudioCaps 的训练

```bash
python -m d25_t6.train \
  --audiocaps \
  --data_path=data \
  --batch_size=4 \
  --batch_size_eval=4 \
  --n_workers=2 \
  --devices=1 \
  --no-compile \
  --no-tau_trainable \
  --seed=492412
```

### 仅下载数据集 (不训练)

```bash
python -m d25_t6.train \
  --no-train \
  --no-test \
  --data_path=data
```

## 重要参数说明

### Mac 特定的参数调整

1. **`--devices=1`**: 使用单个设备 (Mac 通常只有一个 GPU/CPU)
2. **`--no-compile`**: Mac 上不支持 torch.compile，必须禁用
3. **`--n_workers`**:
   - Apple Silicon: 建议 4-8
   - Intel Mac: 建议 2-4
   - 如果遇到问题，可以设置为 0 (单线程)
4. **`--batch_size`**:
   - Apple Silicon: 可以尝试 8-16
   - Intel Mac: 建议 2-4
   - 如果内存不足，减小此值

### 性能优化建议

1. **使用 MPS (Apple Silicon):**

   - PyTorch Lightning 会自动检测并使用 MPS
   - 确保 PyTorch 版本 >= 1.12
2. **内存管理:**

   - 如果遇到内存不足，减小 `batch_size` 和 `batch_size_eval`
   - 考虑使用较小的数据集进行测试
3. **数据加载:**

   - 如果数据加载慢，可以增加 `n_workers`，但不要超过 CPU 核心数
   - 如果遇到多进程问题，设置 `n_workers=0`

## 常见问题

### 1. 导入错误

如果遇到模块导入错误，确保：

```bash
# 在项目根目录下运行
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. MPS 相关错误

如果使用 MPS 时遇到问题，可以强制使用 CPU：

```bash
python -m d25_t6.train --devices=1 --accelerator=cpu ...
```

### 3. 内存不足

减小批次大小：

```bash
python -m d25_t6.train --batch_size=2 --batch_size_eval=2 ...
```

### 4. 数据加载错误

如果遇到数据加载问题，尝试：

```bash
python -m d25_t6.train --n_workers=0 ...
```

## 运行预测

使用训练好的模型进行预测：

```bash
python -m d25_t6.predict \
  --load_ckpt_path=checkpoints/your_checkpoint.ckpt \
  --retrieval_audio_path=path/to/audio/folder \
  --retrieval_captions=path/to/queries.csv \
  --predictions_path=path/to/output/predictions.csv
```

## 性能预期

在 Mac 上的训练速度会比 GPU 慢很多：

- **Apple Silicon (M1/M2/M3)**: 使用 MPS 加速，但仍比 NVIDIA GPU 慢 5-10 倍
- **Intel Mac**: 仅使用 CPU，训练速度会很慢，可能需要数天

建议：

- 先用小数据集测试
- 考虑使用预训练模型进行推理
- 如果可能，使用云 GPU 服务进行训练

## 关于 MPS (Metal Performance Shaders)

Apple Silicon Mac (M1/M2/M3) 支持使用 MPS 进行 GPU 加速。PyTorch Lightning 会自动检测并使用 MPS。

### 验证 MPS 支持

运行以下命令检查 MPS 是否可用：

```bash
python mac_device_helper.py
```

或者：

```python
import torch
print(f"MPS 可用: {torch.backends.mps.is_available()}")
```

### MPS 使用注意事项

1. **自动设备选择**: PyTorch Lightning 会自动使用 MPS，无需手动指定
2. **性能**: MPS 比 CPU 快很多，但仍比 NVIDIA GPU 慢
3. **内存**: MPS 使用系统内存，注意内存使用情况
4. **兼容性**: 某些操作可能不支持 MPS，会自动回退到 CPU

### 如果 MPS 不可用

如果 MPS 不可用或遇到问题，可以强制使用 CPU：

```bash
python -m d25_t6.train --devices=1 --accelerator=cpu ...
```

## 获取帮助

如果遇到问题：

1. 查看原始 README.md 和 README_CN.md
2. 检查 PyTorch 和 PyTorch Lightning 文档
3. 查看项目 Issues: https://github.com/CPJKU/dcase2025_task6_baseline/issues
4. 运行 `python mac_device_helper.py` 检查设备状态
