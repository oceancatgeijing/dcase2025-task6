import os
import pandas as pd
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any, List


class WavCapsLocalDataset(Dataset):
    """
    本地 WavCaps 数据集 - 完全兼容 aac_datasets 接口
    CSV格式: file_path, file_name, category, caption_1, caption_2, caption_3
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        root: Optional[str] = None,
        subset: Optional[str] = None,
        audio_root: Optional[str] = None,
        path_replacement: Optional[Dict[str, str]] = None,
        target_sample_rate: int = 32000,
        max_length: int = 30,
        flat_captions: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        download_audio: bool = False,
        audio_format: str = 'mp3',
        sample_ratio: float = 0.2,  # 新增：采样比例（0.1表示10%）
        seed: Optional[int] = None   # 新增：随机种子（可选，保证可复现）
    ):
        super().__init__()
        
        # 支持 root + subset 初始化（兼容 aac_datasets）
        if csv_path is None and root is not None and subset is not None:
            csv_path = os.path.join(root, f"wavcaps_{subset}_local_label.csv")
        
        if csv_path is None or not os.path.exists(csv_path):
            raise ValueError(f"CSV文件不存在: {csv_path}")
        
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.max_length_samples = target_sample_rate * max_length
        self.transform = transform
        self.flat_captions = flat_captions
        self._eval_mode = False
        self._online_columns = {}
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        print(f"[WavCapsLocal] 读取CSV: {len(df)} 行")
        
        # 路径替换
        if path_replacement:
            for old_str, new_str in path_replacement.items():
                df['file_path'] = df['file_path'].astype(str).str.replace(old_str, new_str, regex=False)
        
        # 构建样本列表
        self.samples = []
        missing_files = []
        
        for idx, row in df.iterrows():
            file_path = row['file_path']
            
            # 如果路径不存在，尝试基于 audio_root 拼接
            if not os.path.exists(file_path) and audio_root is not None:
                category = row.get('category', '')
                file_name = row['file_name']
                alt_path = os.path.join(audio_root, category, file_name)
                if os.path.exists(alt_path):
                    file_path = alt_path
            
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue
            
            # 收集所有有效描述
            captions = []
            for cap_col in ['caption_1', 'caption_2', 'caption_3']:
                if cap_col in row and pd.notna(row[cap_col]) and str(row[cap_col]).strip():
                    captions.append(str(row[cap_col]).strip())
            
            if not captions:
                continue
            
            category = row.get('category', 'unknown')
            file_name = row['file_name']
            
            if flat_captions:
                # 每个caption作为一个独立样本
                for caption in captions:
                    self.samples.append({
                        'file_path': file_path,
                        'captions': caption,
                        'fname': file_name,
                        'category': category,
                        'dataset': 'wavcaps_local',
                        'subset': category,
                    })
            else:
                # 保持为一个样本
                self.samples.append({
                    'file_path': file_path,
                    'captions': captions,
                    'fname': file_name,
                    'category': category,
                    'dataset': 'wavcaps_local',
                    'subset': category,
                })
        
        if missing_files:
            print(f"[WavCapsLocal] 警告: {len(missing_files)} 个文件缺失（仅显示前5个）:")
            for f in missing_files[:5]:
                print(f"  - {os.path.basename(f)}")
        # 随机子采样（如果 sample_ratio < 1.0）
        if sample_ratio < 1.0 and len(self.samples) > 0:
            if seed is not None:
                random.seed(seed)  # 设置随机种子（保证可复现）
            
            # 计算目标数量（至少保留1个样本）
            target_size = max(1, int(len(self.samples) * sample_ratio))
            
            # 随机采样（不重复）
            self.samples = random.sample(self.samples, target_size)
            
            print(f"[WavCapsLocal] 随机采样 {sample_ratio*100:.1f}%:{len(self.samples)}/{len(self.samples)} 个样本保留")
        print(f"[WavCapsLocal] 成功加载 {len(self.samples)} 个有效样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回格式与 aac_datasets (Clotho) 完全兼容
        """
        # 支持元组索引 dataset[i, 'col']
        if isinstance(idx, tuple):
            return self.at(idx[0], idx[1])
        
        sample = self.samples[idx]
        
        # 检查是否有 online column 覆盖（由 custom_loading 设置）
        if 'audio' in self._online_columns:
            waveform = self._online_columns['audio'][0](self, idx)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        else:
            # 默认加载逻辑
            audio_path = sample['file_path']
            try:
                waveform, sr = torchaudio.load(audio_path)
                
                # 重采样
                if sr != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                    waveform = resampler(waveform)
                
                # 单声道（保持维度 [1, T]）
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # 截断或填充
                if waveform.shape[1] > self.max_length_samples:
                    waveform = waveform[:, :self.max_length_samples]
                else:
                    padding = self.max_length_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                    
            except Exception as e:
                print(f"[WavCapsLocal] 加载失败 {audio_path}: {e}")
                waveform = torch.zeros(1, self.max_length_samples)
        
        # ===== 关键修复：兼容 custom_loading 的 custom_transform =====
        # custom_transform 期望接收包含 'audio' 和 'sr' 的字典
        if self.transform:
            temp_sample = {
                'audio': waveform,
                'sr': self.target_sample_rate,
                'duration': waveform.shape[1] / self.target_sample_rate
            }
            temp_sample = self.transform(temp_sample)
            waveform = temp_sample['audio']
        # ============================================================
        
        # 处理 caption
        raw_captions = sample['captions']
        if self.flat_captions:
            caption = raw_captions
        else:
            if isinstance(raw_captions, list):
                if self._eval_mode:
                    caption = raw_captions[0]
                else:
                    caption = random.choice(raw_captions)
            else:
                caption = raw_captions
        
        # 包装为 [[str]] 格式，与 Clotho 一致
        captions_list = [[caption]]
        
        return {
            'audio': waveform,              # [1, T]
            'captions': captions_list,      # [[str]]
            'fname': sample['fname'],
            'fpath': sample['file_path'],
            'dataset': sample['dataset'],
            'subset': sample['subset'],
            'duration': self.max_length_samples / self.target_sample_rate
        }
    
    def at(self, index: int, column: str):
        """兼容 aac_datasets 的 at 方法"""
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Index {index} out of range [0, {len(self.samples)})")
        
        sample = self.samples[index]
        
        mapping = {
            'fpath': 'file_path',
            'fname': 'fname',
            'caption': 'captions',
            'subset': 'subset',
            'dataset': 'dataset'
        }
        
        key = mapping.get(column, column)
        value = sample.get(key)
        
        if column == 'caption' and isinstance(value, list):
            return value[0] if value else ''
        
        return value
    
    @property
    def raw_data(self) -> Dict[str, List]:
        """兼容 aac_datasets 的 raw_data 属性"""
        return {
            'fpath': [s['file_path'] for s in self.samples],
            'fname': [s['fname'] for s in self.samples],
            'subset': [s['subset'] for s in self.samples],
            'dataset': [s['dataset'] for s in self.samples],
        }
    
    def add_online_column(self, name: str, func: Callable, persistent: bool = False):
        """兼容 aac_datasets 的 add_online_column 接口"""
        self._online_columns[name] = (func, persistent)
    
    def set_eval_mode(self, eval_mode: bool = True):
        """设置评估模式"""
        self._eval_mode = eval_mode
    
    def get_broken_files(self, force_refresh: bool = False, broken_files_file: str = "broken_wavcaps_local.json"):
        """便利方法：检查损坏文件"""
        from utils import get_broken_wavcaps_files
        return get_broken_wavcaps_files([self], force_refresh, broken_files_file)