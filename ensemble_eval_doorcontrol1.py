import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import sys
import types
from dataclasses import dataclass

# --- 1. 基础环境补丁 ---
@dataclass
class FakeAudioMetaData:
    sample_rate: int; num_frames: int; num_channels: int; bits_per_sample: int; encoding: str

backend = types.ModuleType("torchaudio.backend")
common = types.ModuleType("torchaudio.backend.common")
common.AudioMetaData = FakeAudioMetaData
backend.common = common
sys.modules["torchaudio.backend"] = backend
sys.modules["torchaudio.backend.common"] = common

from d25_t6.retrieval_module import AudioRetrievalModel
from d25_t6.datasets.audio_loading import custom_loading
from aac_datasets import Clotho
from d25_t6.datasets.batch_collate import CustomCollate

# --- 2. 基于温度校准的门控网络 ---
class TemperatureCalibratedGate(nn.Module):
    """
    创新点：基于模型校准理论（Temperature Scaling）的自适应融合
    核心思想：
    1. 每路模型学习一个温度参数 T，用于校准分数的置信度
    2. 温度 T > 1 表示模型过自信（分数太尖锐），需要平滑
    3. 温度 T < 1 表示模型不自信（分数太平坦），需要锐化
    4. 校准后再进行可学习的加权融合
    
    理论支撑：Platt Scaling, Temperature Scaling (Guo et al., 2017)
    优势：参数量极少（3个参数），可解释性强，不易过拟合
    """
    def __init__(self, init_temp_bi=1.0, init_temp_trans=1.0, init_alpha=0.5):
        super(TemperatureCalibratedGate, self).__init__()
        
        # 可学习的温度参数（使用log空间保证正数）
        self.log_temp_bi = nn.Parameter(torch.log(torch.tensor(init_temp_bi)))
        self.log_temp_trans = nn.Parameter(torch.log(torch.tensor(init_temp_trans)))
        
        # 可学习的融合权重（logit空间）
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(init_alpha)))
        
        # 记录训练过程中的参数变化
        self.history = {
            'temp_bi': [],
            'temp_trans': [],
            'alpha': []
        }
    
    @property
    def temp_bi(self):
        """Bi-Encoder的温度参数，保证 > 0"""
        return torch.exp(self.log_temp_bi)
    
    @property
    def temp_trans(self):
        """Transformer的温度参数，保证 > 0"""
        return torch.exp(self.log_temp_trans)
    
    @property
    def alpha(self):
        """融合权重，范围 [0.2, 0.8]"""
        # sigmoid映射到(0,1)，再缩放到[0.2, 0.8]
        return 0.2 + 0.6 * torch.sigmoid(self.logit_alpha)
    
    def forward(self, s_bi, s_trans, return_params=False):
        """
        前向传播：温度校准 + 加权融合
        
        Args:
            s_bi: Bi-Encoder的相似度分数，形状 (B,) 或 (B, B)
            s_trans: Transformer的相似度分数，形状与s_bi相同
            return_params: 是否返回当前参数值
            
        Returns:
            fused_score: 融合后的分数
            alpha: 当前融合权重（如果return_params=True）
        """
        # 温度校准：分数除以温度（相当于softmax中的温度缩放）
        # T > 1: 分数变得更平滑（降低过自信模型的影响力）
        # T < 1: 分数变得更尖锐（提升不自信模型的区分度）
        s_bi_cal = s_bi / self.temp_bi
        s_trans_cal = s_trans / self.temp_trans
        
        # 加权融合
        alpha = self.alpha
        fused_score = alpha * s_bi_cal + (1 - alpha) * s_trans_cal
        
        if return_params:
            return fused_score, alpha, self.temp_bi, self.temp_trans
        return fused_score
    
    def record_params(self):
        """记录当前参数值"""
        self.history['temp_bi'].append(self.temp_bi.item())
        self.history['temp_trans'].append(self.temp_trans.item())
        self.history['alpha'].append(self.alpha.item())
    
    def get_param_summary(self):
        """获取参数变化摘要"""
        return {
            'temp_bi': f"{self.temp_bi.item():.4f}",
            'temp_trans': f"{self.temp_trans.item():.4f}",
            'alpha': f"{self.alpha.item():.4f}",
            'temp_ratio': f"{(self.temp_bi / self.temp_trans).item():.4f}"
        }

# --- 3. 辅助函数 ---
def _batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# --- 4. 训练温度校准门控 ---
def train_temperature_gate(model_bi, model_trans, data_path='data', batch_size=32, epochs=10):
    """
    训练基于温度校准的门控网络
    
    优化目标：使用InfoNCE损失，直接优化融合后的检索性能
    关键：温度参数和融合权重联合优化
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = custom_loading(Clotho(subset="dev", root=data_path, flat_captions=True))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             collate_fn=CustomCollate(), drop_last=True)

    gate_net = TemperatureCalibratedGate().to(device)
    
    # 使用较大的学习率，因为参数量很少
    optimizer = torch.optim.Adam(gate_net.parameters(), lr=0.01)
    
    # 学习率调度：余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
    
    model_bi.eval()
    model_trans.eval()

    print("\n" + "="*60)
    print("Training Temperature-Calibrated Gate")
    print("="*60)
    print(f"Initial params: {gate_net.get_param_summary()}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("-"*60)

    best_loss = float('inf')
    best_params = None

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = _batch_to_device(batch, device)
            B = batch['audio'].size(0)
            
            optimizer.zero_grad()

            with torch.no_grad():
                # 提取特征
                a_bi = model_bi.forward_audio(batch)  # (B, 1024)
                t_bi = model_bi.forward_text(batch)   # (B, 1024)
                
                # Bi-Encoder相似度矩阵
                s_bi = torch.matmul(t_bi, a_bi.T)     # (B, B)
                
                # Transformer相似度矩阵
                try:
                    # 尝试批量计算
                    a_exp = a_bi.unsqueeze(1).expand(-1, B, -1)  # (B, B, 1024)
                    t_exp = t_bi.unsqueeze(0).expand(B, -1, -1)  # (B, B, 1024)
                    s_trans = model_trans.forward_fusion(a_exp, t_exp)
                    
                    # 确保维度正确
                    if s_trans.dim() == 1:
                        s_trans = torch.diag_embed(s_trans)
                    elif s_trans.shape != (B, B):
                        if s_trans.numel() == B * B:
                            s_trans = s_trans.view(B, B)
                        else:
                            raise ValueError(f"Unexpected shape: {s_trans.shape}")
                            
                except Exception as e:
                    # 回退到循环计算
                    s_trans = torch.zeros(B, B, device=device)
                    for i in range(B):
                        s_trans[i] = model_trans.forward_fusion(a_bi, t_bi[i:i+1].expand(B, -1))

            # 门控前向：温度校准 + 融合
            fused_sim, alpha, temp_bi, temp_trans = gate_net(s_bi, s_trans, return_params=True)
            
            # InfoNCE损失：优化融合后的相似度矩阵
            # 对角线是正样本
            labels = torch.arange(B, device=device)
            
            # 温度缩放后的对比学习（注意：这里的温度与门控的温度不同）
            temperature = 0.07
            loss = F.cross_entropy(fused_sim / temperature, labels)
            
            # 辅助损失：鼓励温度参数有区分度（正则化）
            # 避免两个温度都学到相同的值
            temp_diff_penalty = -0.001 * torch.abs(temp_bi - temp_trans)
            loss = loss + temp_diff_penalty

            loss.backward()
            
            # 梯度裁剪（防止温度参数变化过快）
            torch.nn.utils.clip_grad_norm_(gate_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 记录参数
        gate_net.record_params()
        
        avg_loss = total_loss / num_batches
        current_params = gate_net.get_param_summary()
        
        print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")
        print(f"  Params: T_bi={current_params['temp_bi']}, "
              f"T_trans={current_params['temp_trans']}, "
              f"alpha={current_params['alpha']}, "
              f"ratio={current_params['temp_ratio']}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {k: v.item() if isinstance(v, torch.Tensor) else v 
                          for k, v in gate_net.state_dict().items()}
            torch.save(gate_net.state_dict(), "temperature_gate_best.pth")
            print(f"  [*] Saved best model (loss: {best_loss:.4f})")

    # 加载最佳模型
    print("\n" + "-"*60)
    print("Training completed. Loading best model...")
    gate_net.load_state_dict(torch.load("temperature_gate_best.pth"))
    
    # 打印参数变化轨迹
    print("\nParameter evolution:")
    print(f"  Temp_bi:    {gate_net.history['temp_bi'][0]:.4f} -> {gate_net.history['temp_bi'][-1]:.4f}")
    print(f"  Temp_trans: {gate_net.history['temp_trans'][0]:.4f} -> {gate_net.history['temp_trans'][-1]:.4f}")
    print(f"  Alpha:      {gate_net.history['alpha'][0]:.4f} -> {gate_net.history['alpha'][-1]:.4f}")
    print("="*60)
    
    return gate_net

# --- 5. 推理函数 ---
def run_temperature_gate_eval(bi_ckpt_path, trans_ckpt_path, data_path='data'):
    """
    使用温度校准门控进行推理评估
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print("Loading models...")
    model_bi = AudioRetrievalModel.load_from_checkpoint(bi_ckpt_path).to(device).eval()
    model_trans = AudioRetrievalModel.load_from_checkpoint(trans_ckpt_path).to(device).eval()
    
    # 加载门控网络
    gate_net = TemperatureCalibratedGate().to(device)
    if os.path.exists("temperature_gate_best.pth"):
        gate_net.load_state_dict(torch.load("temperature_gate_best.pth"))
        print("Successfully loaded temperature_gate_best.pth")
        print(f"Loaded params: {gate_net.get_param_summary()}")
    else:
        print("Warning: No trained gate found, using default initialization")
        print(f"Default params: {gate_net.get_param_summary()}")
    gate_net.eval()

    # 准备数据
    test_ds = custom_loading(Clotho(subset="eval", root=data_path, flat_captions=True))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=CustomCollate())

    print("\nExtracting embeddings...")
    all_a_bi, all_t_bi, all_a_trans, all_t_trans = [], [], [], []
    all_fnames, all_captions = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Feature extraction"):
            batch = _batch_to_device(batch, device)
            all_a_bi.append(model_bi.forward_audio(batch).cpu())
            all_t_bi.append(model_bi.forward_text(batch).cpu())
            
            a_tr, t_tr = model_trans.forward(batch)
            all_a_trans.append(a_tr.cpu())
            all_t_trans.append(t_tr.cpu())
            
            all_fnames.extend(batch['fname'])
            all_captions.extend([c[0] if isinstance(c, list) else c for c in batch['captions']])

    # 特征准备
    audio_bi = torch.cat(all_a_bi).to(device)
    text_bi = torch.cat(all_t_bi).to(device)
    audio_tr = torch.cat(all_a_trans).to(device)
    text_tr = torch.cat(all_t_trans).to(device)

    # 去重
    unique_paths, select_indices, targets, path_to_idx = [], [], [], {}
    for i, p in enumerate(all_fnames):
        if p not in path_to_idx:
            path_to_idx[p] = len(unique_paths)
            unique_paths.append(p)
            select_indices.append(i)
        targets.append(path_to_idx[p])
    
    u_audio_bi = audio_bi[select_indices]
    u_audio_tr = audio_tr[select_indices]
    num_texts, num_audios = text_bi.shape[0], u_audio_bi.shape[0]
    
    final_scores = np.zeros((num_texts, num_audios))
    all_alphas = []
    
    # 统计
    both_correct = 0
    only_bi_correct = 0
    only_tr_correct = 0
    both_wrong = 0

    print(f"\nTemperature-calibrated fusion (texts: {num_texts}, audios: {num_audios})...")
    
    # 获取固定的温度参数和alpha（推理时不更新）
    with torch.no_grad():
        fixed_temp_bi = gate_net.temp_bi.item()
        fixed_temp_trans = gate_net.temp_trans.item()
        fixed_alpha = gate_net.alpha.item()
        
    print(f"Using fixed params: T_bi={fixed_temp_bi:.4f}, T_trans={fixed_temp_trans:.4f}, alpha={fixed_alpha:.4f}")

    with torch.no_grad():
        for i in tqdm(range(num_texts), desc="Fusion"):
            # 计算相似度
            s_bi = torch.matmul(text_bi[i], u_audio_bi.T)  # (num_audios,)
            s_tr = model_trans.forward_fusion(u_audio_tr, text_tr[i].expand(num_audios, -1))
            
            # 错误分析（前500样本）
            if i < 500:
                target = targets[i]
                bi_rank = torch.argmax(s_bi).item()
                tr_rank = torch.argmax(s_tr).item()
                
                bi_correct = (bi_rank == target)
                tr_correct = (tr_rank == target)
                
                if bi_correct and tr_correct:
                    both_correct += 1
                elif bi_correct and not tr_correct:
                    only_bi_correct += 1
                elif not bi_correct and tr_correct:
                    only_tr_correct += 1
                else:
                    both_wrong += 1
            
            # 温度校准（使用固定的温度参数）
            s_bi_cal = s_bi / fixed_temp_bi
            s_tr_cal = s_tr / fixed_temp_trans
            
            # 融合（使用固定的alpha）
            alpha = fixed_alpha
            f_score = alpha * s_bi_cal + (1 - alpha) * s_tr_cal
            
            all_alphas.append(alpha)
            final_scores[i] = f_score.cpu().numpy()

    # 打印分析结果
    total_analyzed = both_correct + only_bi_correct + only_tr_correct + both_wrong
    if total_analyzed > 0:
        print(f"\n{'='*60}")
        print("错误互补性分析 (前 500 个样本):")
        print(f"  Both correct:    {both_correct:3d} ({both_correct/total_analyzed*100:5.1f}%)")
        print(f"  Only Bi correct: {only_bi_correct:3d} ({only_bi_correct/total_analyzed*100:5.1f}%)")
        print(f"  Only Tr correct: {only_tr_correct:3d} ({only_tr_correct/total_analyzed*100:5.1f}%)")
        print(f"  Both wrong:      {both_wrong:3d} ({both_wrong/total_analyzed*100:5.1f}%)")
        print(f"  互补潜力: {(only_bi_correct + only_tr_correct)/total_analyzed*100:.1f}%")
        print(f"{'='*60}")
    
    # Alpha统计（应该是常数）
    print(f"\n{'='*60}")
    print("门控权重 (Alpha) 统计信息:")
    print(f"  平均值 (Mean):  {np.mean(all_alphas):.4f}")
    print(f"  最小值 (Min):   {np.min(all_alphas):.4f}")
    print(f"  最大值 (Max):   {np.max(all_alphas):.4f}")
    print(f"  标准差 (Std):   {np.std(all_alphas):.4f}")
    print(f"  (注：温度校准门控使用全局固定alpha)")
    print(f"{'='*60}")
    
    # 计算指标
    print("\n>>> Calculating metrics...")
    targets = np.array(targets)
    top_10 = np.argsort(-final_scores, axis=1)[:, :10]

    r1 = (top_10[:, :1] == targets[:, None]).any(axis=1).mean()
    r5 = (top_10[:, :5] == targets[:, None]).any(axis=1).mean()
    r10 = (top_10[:, :10] == targets[:, None]).any(axis=1).mean()

    aps = []
    for i in range(len(targets)):
        rank = np.where(top_10[i] == targets[i])[0]
        aps.append(1.0 / (rank[0] + 1) if len(rank) > 0 else 0)
    mAP_std = np.mean(aps)

    print(f"\n{'='*60}")
    print("Temperature-Calibrated Gate Results:")
    print(f"  R@1:  {r1:.4f}")
    print(f"  R@5:  {r5:.4f}")
    print(f"  R@10: {r10:.4f}")
    print(f"  mAP@10 (Std): {mAP_std:.4f}")
    print(f"{'='*60}")

    # 多正样本mAP
    metadata_path = "resources/metadata_eval.csv"
    if os.path.exists(metadata_path):
        print(f"\nLoading metadata: {metadata_path}")
        try:
            matched_files = pd.read_csv(metadata_path)
            
            def get_query_idx(q):
                try:
                    return all_captions.index(q)
                except ValueError:
                    return -1

            def get_audio_indices(paths):
                indices = []
                for p in paths:
                    try:
                        indices.append(unique_paths.index(p))
                    except ValueError:
                        continue
                return indices

            def safe_eval(x):
                try: 
                    return ast.literal_eval(x) if isinstance(x, str) else x
                except: 
                    return []

            matched_files["audio_filenames"] = matched_files["audio_filenames"].apply(safe_eval)
            matched_files["query_index"] = matched_files["query"].apply(get_query_idx)
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].apply(get_audio_indices)

            valid_matched = matched_files[matched_files["query_index"] != -1].copy()
            valid_matched = valid_matched[valid_matched["new_audio_indices"].map(len) > 0]
            
            print(f"Valid entries: {len(valid_matched)}/{len(matched_files)}")

            def calculate_ap_multi(row):
                scores_row = final_scores[row["query_index"]]
                rel_indices = row["new_audio_indices"]
                sort_indices = np.argsort(-scores_row)
                
                ranks = sorted([np.where(sort_indices == rel_idx)[0][0] for rel_idx in rel_indices])
                ap = 0.0
                for i, rank in enumerate(ranks):
                    if rank < 10:
                        ap += (i + 1) / (rank + 1)
                return ap / len(rel_indices)

            mAP_multi = valid_matched.apply(calculate_ap_multi, axis=1).mean()
            print(f"\nDCASE Official Metric:")
            print(f"  test_multiple_positives/mAP@10: {mAP_multi:.4f}")
            print(f"{'='*60}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print(f"Metadata not found: {metadata_path}")

# --- 6. 主程序 ---
if __name__ == "__main__":
    BI_CKPT = "checkpoints/cosmic-paper-15/epoch=13.ckpt"
    TRANS_CKPT = "checkpoints/revived-snowball-16/epoch=19.ckpt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练模式
    print(">>> Training mode")
    m_bi = AudioRetrievalModel.load_from_checkpoint(BI_CKPT).to(device)
    m_tr = AudioRetrievalModel.load_from_checkpoint(TRANS_CKPT).to(device)
    my_gate = train_temperature_gate(m_bi, m_tr, epochs=5, batch_size=32)
    
    # 推理模式（注释掉训练后运行）
    # print(">>> Evaluation mode")
    # run_temperature_gate_eval(BI_CKPT, TRANS_CKPT)