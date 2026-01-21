# Clotho v2.1 数据集手动下载指南

由于自动下载可能因网络问题失败，您可以手动下载 Clotho v2.1 数据集。

## 方法一：使用浏览器或下载工具下载

### 需要下载的文件

根据训练脚本，您需要下载以下文件（dev, val, eval 三个子集）：

#### Development 集（开发集）
1. **音频文件** (约 4.23 GB):
   - URL: https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1
   - 保存为: `data/CLOTHO_v2.1/archives/clotho_audio_development.7z`
   - MD5 校验和: `c8b05bc7acdb13895bb3c6a29608667e`

2. **字幕文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_captions_development.csv`

3. **元数据文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_metadata_development.csv`

#### Validation 集（验证集）
1. **音频文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1
   - 保存为: `data/CLOTHO_v2.1/archives/clotho_audio_validation.7z`

2. **字幕文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_captions_validation.csv`

3. **元数据文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_metadata_validation.csv`

#### Evaluation 集（评估集）
1. **音频文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1
   - 保存为: `data/CLOTHO_v2.1/archives/clotho_audio_evaluation.7z`

2. **字幕文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_captions_evaluation.csv`

3. **元数据文件**:
   - URL: https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1
   - 保存为: `data/CLOTHO_v2.1/csv_files/clotho_metadata_evaluation.csv`

### 下载步骤

1. **创建目录结构**:
   ```bash
   cd /Users/seki/Desktop/毕业设计/dcase2025_task6_baseline
   mkdir -p data/CLOTHO_v2.1/archives
   mkdir -p data/CLOTHO_v2.1/csv_files
   ```

2. **使用支持断点续传的工具下载**（推荐）:
   
   **使用 wget** (如果已安装):
   ```bash
   cd data/CLOTHO_v2.1/archives
   wget -c https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1 -O clotho_audio_development.7z
   wget -c https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1 -O clotho_audio_validation.7z
   wget -c https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1 -O clotho_audio_evaluation.7z
   
   cd ../csv_files
   wget -c https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1 -O clotho_captions_development.csv
   wget -c https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1 -O clotho_metadata_development.csv
   wget -c https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1 -O clotho_captions_validation.csv
   wget -c https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1 -O clotho_metadata_validation.csv
   wget -c https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1 -O clotho_captions_evaluation.csv
   wget -c https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1 -O clotho_metadata_evaluation.csv
   ```
   
   **使用 curl**:
   ```bash
   cd data/CLOTHO_v2.1/archives
   curl -L -C - -o clotho_audio_development.7z "https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1"
   curl -L -C - -o clotho_audio_validation.7z "https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1"
   curl -L -C - -o clotho_audio_evaluation.7z "https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1"
   ```

3. **使用浏览器下载**:
   - 访问 https://zenodo.org/record/4783391
   - 点击每个文件的下载链接
   - 将文件保存到对应的目录

4. **验证文件完整性**:
   ```bash
   cd /Users/seki/Desktop/毕业设计/dcase2025_task6_baseline
   source ../.venv/bin/activate
   python -c "
   import hashlib
   def md5(fname):
       hash_md5 = hashlib.md5()
       with open(fname, 'rb') as f:
           for chunk in iter(lambda: f.read(4096), b''):
               hash_md5.update(chunk)
       return hash_md5.hexdigest()
   
   expected = 'c8b05bc7acdb13895bb3c6a29608667e'
   actual = md5('data/CLOTHO_v2.1/archives/clotho_audio_development.7z')
   print(f'Development 音频文件 MD5: {actual}')
   print(f'期望值: {expected}')
   print(f'匹配: {actual == expected}')
   "
   ```

5. **运行训练脚本自动解压**:
   下载完成后，运行训练脚本会自动解压文件：
   ```bash
   python -m d25_t6.train --no-train --no-test --data_path=data
   ```

## 方法二：使用 Python 脚本下载（支持断点续传）

运行提供的下载脚本（见 `scripts/download_clotho_manual.py`）

## 注意事项

1. **网络稳定性**: 如果网络不稳定，建议使用支持断点续传的工具（wget, curl, 或下载管理器）
2. **磁盘空间**: 确保有足够的磁盘空间（至少 25GB）
3. **下载时间**: 根据网络速度，完整下载可能需要数小时
4. **文件完整性**: 下载后务必验证 MD5 校验和

## 如果下载仍然失败

1. 尝试使用 VPN 或更换网络
2. 使用下载管理器（如 IDM, Aria2 等）
3. 分时段下载（避开网络高峰期）
4. 考虑使用云服务器下载后传输到本地

