#!/usr/bin/env python3
"""
手动下载 Clotho v2.1 数据集的脚本
支持断点续传和重试机制
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# Clotho v2.1 下载链接和 MD5 校验和
CLOTHO_V2_1_FILES = {
    "archives": {
        "clotho_audio_development.7z": {
            "url": "https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1",
            "md5": "c8b05bc7acdb13895bb3c6a29608667e"
        },
        "clotho_audio_validation.7z": {
            "url": "https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1",
            "md5": "7dba730be08bada48bd15dc4e668df59"
        },
        "clotho_audio_evaluation.7z": {
            "url": "https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1",
            "md5": "4569624ccadf96223f19cb59fe4f849f"
        }
    },
    "csv_files": {
        "clotho_captions_development.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1",
            "md5": "d4090b39ce9f2491908eebf4d5b09bae"
        },
        "clotho_metadata_development.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1",
            "md5": "170d20935ecfdf161ce1bb154118cda5"
        },
        "clotho_captions_validation.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1",
            "md5": "5879e023032b22a2c930aaa0528bead4"
        },
        "clotho_metadata_validation.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1",
            "md5": "2e010427c56b1ce6008b0f03f41048ce"
        },
        "clotho_captions_evaluation.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1",
            "md5": "1b16b9e57cf7bdb7f13a13802aeb57e2"
        },
        "clotho_metadata_evaluation.csv": {
            "url": "https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1",
            "md5": "13946f054d4e1bf48079813aac61bf77"
        }
    }
}


def md5_hash(filepath):
    """计算文件的 MD5 哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None


def download_file(url, filepath, expected_md5=None, max_retries=3, chunk_size=8192):
    """下载文件，支持断点续传和重试"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否已存在且完整
    if filepath.exists():
        if expected_md5:
            actual_md5 = md5_hash(filepath)
            if actual_md5 == expected_md5:
                print(f"✓ {filepath.name} 已存在且校验通过，跳过下载")
                return True
            else:
                print(f"✗ {filepath.name} 已存在但校验失败，将重新下载")
                filepath.unlink()
        else:
            print(f"✓ {filepath.name} 已存在，跳过下载")
            return True
    
    # 获取已下载的文件大小（用于断点续传）
    resume_pos = filepath.stat().st_size if filepath.exists() else 0
    
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
        print(f"从 {resume_pos} 字节处继续下载 {filepath.name}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # 处理断点续传
            if response.status_code == 206:  # Partial Content
                mode = 'ab'
            elif response.status_code == 200:
                mode = 'wb'
                resume_pos = 0
            else:
                print(f"✗ 下载失败: HTTP {response.status_code}")
                return False
            
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
            
            with open(filepath, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit='B',
                    unit_scale=True,
                    desc=filepath.name,
                    leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 验证文件
            if expected_md5:
                actual_md5 = md5_hash(filepath)
                if actual_md5 != expected_md5:
                    print(f"✗ {filepath.name} MD5 校验失败 (期望: {expected_md5}, 实际: {actual_md5})")
                    if attempt < max_retries - 1:
                        print(f"  重试 {attempt + 1}/{max_retries}...")
                        filepath.unlink()
                        continue
                    return False
                else:
                    print(f"✓ {filepath.name} 下载完成且校验通过")
            else:
                print(f"✓ {filepath.name} 下载完成")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ 下载 {filepath.name} 时出错: {e}")
            if attempt < max_retries - 1:
                print(f"  重试 {attempt + 1}/{max_retries}...")
                if filepath.exists() and filepath.stat().st_size == 0:
                    filepath.unlink()
            else:
                print(f"✗ {filepath.name} 下载失败，已重试 {max_retries} 次")
                return False
    
    return False


def main():
    """主函数"""
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data"
    
    base_path = Path(data_path) / "CLOTHO_v2.1"
    archives_path = base_path / "archives"
    csv_path = base_path / "csv_files"
    
    print("=" * 60)
    print("Clotho v2.1 数据集手动下载工具")
    print("=" * 60)
    print(f"数据路径: {base_path.absolute()}")
    print()
    
    # 下载压缩文件
    print("下载音频压缩文件...")
    for filename, file_info in CLOTHO_V2_1_FILES["archives"].items():
        filepath = archives_path / filename
        download_file(
            file_info["url"],
            filepath,
            expected_md5=file_info.get("md5")
        )
    
    print()
    print("下载 CSV 文件...")
    for filename, file_info in CLOTHO_V2_1_FILES["csv_files"].items():
        filepath = csv_path / filename
        download_file(
            file_info["url"],
            filepath,
            expected_md5=file_info.get("md5")
        )
    
    print()
    print("=" * 60)
    print("下载完成！")
    print("=" * 60)
    print("现在可以运行训练脚本，它会自动解压文件：")
    print(f"  python -m d25_t6.train --no-train --no-test --data_path={data_path}")


if __name__ == "__main__":
    main()

