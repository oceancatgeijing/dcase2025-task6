import ast
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import sys
import types
from dataclasses import dataclass

# --- 1. 基础环境补丁 (保留你的原始逻辑) ---
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

# --- 2. 动态门控网络定义 ---
class GatedFusionNetwork(nn.Module):
    """
    输入音频和文本的嵌入特征，输出 Bi-Encoder 的权重 alpha。
    alpha 趋向 1 则信任 Bi-Encoder，趋向 0 则信任 Transformer。
    """
    def __init__(self, audio_dim=1024, text_dim=1024):
        super(GatedFusionNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + text_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # --- 核心修改：让最后一层的初始输出接近 0 ---
        # 这样 torch.sigmoid(0) 就会得到 0.5，实现默认等权重
        nn.init.constant_(self.gate[-1].weight, 0)
        nn.init.constant_(self.gate[-1].bias, 0)

    def forward(self, audio_embed, text_embed, s_bi, s_trans):
        # audio_embed: (B, dim), text_embed: (B, dim)
        # 将特征和分数拼接在一起输入，让门控知道每个模型“自认为”的得分
        # 确保输入维度正确
        if s_bi.dim() == 1:
            s_bi = s_bi.unsqueeze(-1)
        if s_trans.dim() == 1:
            s_trans = s_trans.unsqueeze(-1)
            
        # 新增：差值和比值作为额外信号
        diff = s_bi - s_trans
        # 安全 log 比值（避免除零和负数）
        ratio = torch.log((torch.abs(s_bi) + 1e-8) / (torch.abs(s_trans) + 1e-8))
        
        feat = torch.cat([audio_embed, text_embed, s_bi, s_trans, diff, ratio], dim=-1)
        
        out = self.gate(feat)
        # 关键：限制输出范围 [0.1, 0.9]，避免 sigmoid 饱和到 0 或 1
        return 0.1 + 0.8 * torch.sigmoid(out)

# --- 3. 辅助函数 ---
def _batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def min_max_normalize_torch(tensor):
    """在 Torch 张量上进行归一化，保持梯度"""
    t_min = tensor.min()
    t_max = tensor.max()
    return (tensor - t_min) / (t_max - t_min + 1e-8)

# --- 4. 训练门控网络函数 ---
def train_gate(model_bi, model_trans, data_path='data', batch_size=32, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用 Clotho dev set 训练门控，避免 eval set 泄露
    train_ds = custom_loading(Clotho(subset="dev", root=data_path, flat_captions=True))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=CustomCollate())

    gate_net = GatedFusionNetwork(audio_dim=1024, text_dim=1024).to(device)
    optimizer = torch.optim.Adam(gate_net.parameters(), lr=1e-4)
    model_bi.eval(); model_trans.eval()

    print("\n>>> Start Training Gated Fusion Network...")
    for epoch in range(epochs):
        total_loss = 0
        total_alpha = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()

            with torch.no_grad():
                # --- 1. 补充这两行：调用模型提取音频和文本特征 ---
                a_bi = model_bi.forward_audio(batch)  # 提取音频特征 (B, 1024)
                t_bi = model_bi.forward_text(batch)   # 提取文本特征 (B, 1024)
                # 提取两路特征
                s_bi = torch.matmul(t_bi, a_bi.T)
                s_trans_val = model_trans.forward_fusion(a_bi, t_bi) 
                
                # --- 新增：分数标准化 (Z-Score) ---
                # 这样处理后，两组分数都在 0 均值附近，门控才能公平比较
                s_bi_diag = torch.diag(s_bi)
                s_bi_std = (s_bi_diag - s_bi_diag.mean()) / (s_bi_diag.std() + 1e-8)
                s_tr_std = (s_trans_val - s_trans_val.mean()) / (s_trans_val.std() + 1e-8)

            # 动态计算 alpha (针对当前 batch 内            
            try:
                alpha = gate_net(a_bi, t_bi, s_bi_diag, s_trans_val).squeeze()
            except Exception as e:
                print(f"\n=== Gate Net Input Shapes ===")
                print(f"a_bi: {a_bi.shape}, t_bi: {t_bi.shape}")
                print(f"s_bi_diag: {s_bi_diag.shape}, s_trans_val: {s_trans_val.shape}")
                print(f"Error: {e}")
                raise
                

            total_alpha += alpha.mean().item()
            # 融合计算使用标准化后的分数，这样 Loss 梯度更稳定
            fused_score = alpha * s_bi_std + (1 - alpha) * s_tr_std
            loss = -fused_score.mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_alpha = total_alpha / len(train_loader) # 计算 Epoch 级别的平均权重
        
        # print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Avg Alpha: {avg_alpha:.4f}")
    
    torch.save(gate_net.state_dict(), "gated_fusion_weights.pth")
    return gate_net

# --- 5. 核心推理函数 (集成门控) ---
def run_gated_eval(bi_ckpt_path, trans_ckpt_path, data_path='data'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model_bi = AudioRetrievalModel.load_from_checkpoint(bi_ckpt_path).to(device).eval()
    model_trans = AudioRetrievalModel.load_from_checkpoint(trans_ckpt_path).to(device).eval()
    
    gate_net = GatedFusionNetwork(audio_dim=1024, text_dim=1024).to(device)
    if os.path.exists("gated_fusion_weights.pth"):
        gate_net.load_state_dict(torch.load("gated_fusion_weights.pth"))
        print("Successfully loaded gated_fusion_weights.pth")
    gate_net.eval()

    # 2. 准备数据
    test_ds = custom_loading(Clotho(subset="eval", root=data_path, flat_captions=True))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=CustomCollate())

    print("Extracting Embeddings & Gating...")
    all_a_bi, all_t_bi, all_a_trans, all_t_trans = [], [], [], []
    all_fnames, all_captions = [], [] # 在此处定义，确保作用域覆盖后续逻辑

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = _batch_to_device(batch, device)
            all_a_bi.append(model_bi.forward_audio(batch).cpu())
            all_t_bi.append(model_bi.forward_text(batch).cpu())
            
            a_tr, t_tr = model_trans.forward(batch)
            all_a_trans.append(a_tr.cpu())
            all_t_trans.append(t_tr.cpu())
            
            all_fnames.extend(batch['fname'])
            # 重要：确保提取了所有的文本描述用于后续 mAP 映射
            all_captions.extend([c[0] if isinstance(c, list) else c for c in batch['captions']])

    # 3. 特征准备与去重
    audio_bi = torch.cat(all_a_bi).to(device)
    text_bi = torch.cat(all_t_bi).to(device)
    audio_tr = torch.cat(all_a_trans).to(device)
    text_tr = torch.cat(all_t_trans).to(device)

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
    all_alphas = [] # 新增：记录测试集所有的 alpha

    # 4. 执行门控融合
    print("Dynamic Fusing Matrix...")

    # 新增：错误分析统计
    both_correct = 0
    only_bi_correct = 0
    only_tr_correct = 0
    both_wrong = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_texts)):
            s_bi = torch.matmul(text_bi[i], u_audio_bi.T)
            s_tr = model_trans.forward_fusion(u_audio_tr, text_tr[i].expand(num_audios, -1))

             # 新增：错误分析（只在第一次迭代或采样部分样本）
            if i < 500:  # 分析前 100 个样本，避免太慢
                target = targets[i]
                bi_rank = torch.argsort(-s_bi)[0].item()
                tr_rank = torch.argsort(-s_tr)[0].item()
                
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
                    
            # --- 关键修改：改用 Z-Score 标准化 ---
            s_bi_norm = (s_bi - s_bi.mean()) / (s_bi.std() + 1e-8)
            s_tr_norm = (s_tr - s_tr.mean()) / (s_tr.std() + 1e-8)

            # --- 关键修改：门控同时接收 Embedding 和标准化后的 Score ---
            alpha = gate_net(u_audio_bi, text_bi[i].expand(num_audios, -1), 
                 s_bi_norm, s_tr_norm).squeeze()
            
            all_alphas.append(alpha.cpu().numpy())
            
            # 最终得分融合
            f_score = alpha * s_bi_norm + (1 - alpha) * s_tr_norm
            final_scores[i] = f_score.cpu().numpy()


    # 新增：打印错误分析结果
    total_analyzed = both_correct + only_bi_correct + only_tr_correct + both_wrong
    if total_analyzed > 0:
        print(f"\n" + "="*40)
        print(f"错误互补性分析 (前 {total_analyzed} 个样本):")
        print(f"Both correct:   {both_correct} ({both_correct/total_analyzed*100:.1f}%)")
        print(f"Only Bi correct: {only_bi_correct} ({only_bi_correct/total_analyzed*100:.1f}%)")
        print(f"Only Tr correct: {only_tr_correct} ({only_tr_correct/total_analyzed*100:.1f}%)")
        print(f"Both wrong:     {both_wrong} ({both_wrong/total_analyzed*100:.1f}%)")
        print(f"互补潜力: {(only_bi_correct + only_tr_correct)/total_analyzed*100:.1f}%")
        print("="*40)
        
    # --- 评估结束后打印统计信息 ---
    all_alphas = np.concatenate(all_alphas)
    print(f"\n" + "="*30)
    print(f"门控权重 (Alpha) 统计信息:")
    print(f"平均值 (Mean): {np.mean(all_alphas):.4f}")
    print(f"最小值 (Min):  {np.min(all_alphas):.4f}")
    print(f"最大值 (Max):  {np.max(all_alphas):.4f}")
    print(f"标准差 (Std):  {np.std(all_alphas):.4f}")
    print("="*30)
    
    # 5. 指标计算 (Recall & Standard mAP)
    print("\n>>> Calculating Basic Metrics...")
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

    print(f"{'='*30}\nBasic Metrics:\nR@1: {r1:.4f} | R@5: {r5:.4f} | R@10: {r10:.4f}\nmAP@10 (Std): {mAP_std:.4f}\n{'='*30}")

    # 6. 多正样本指标计算 (解决 NameError 的关键位置)
    metadata_path = "resources/metadata_eval.csv"
    if os.path.exists(metadata_path):
        print(f"Loading metadata for multi-positive mAP: {metadata_path}")
        try:
            matched_files = pd.read_csv(metadata_path)
            
            # --- 内部辅助函数，解决作用域问题 ---
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
                try: return ast.literal_eval(x) if isinstance(x, str) else x
                except: return []

            # 执行映射
            matched_files["audio_filenames"] = matched_files["audio_filenames"].apply(safe_eval)
            matched_files["query_index"] = matched_files["query"].apply(get_query_idx)
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].apply(get_audio_indices)

            # 过滤
            valid_matched = matched_files[matched_files["query_index"] != -1].copy()
            valid_matched = valid_matched[valid_matched["new_audio_indices"].map(len) > 0]
            
            print(f"Metadata matched: {len(valid_matched)}/{len(matched_files)} valid entries.")

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
            print(f"DCASE Official Metric:\ntest_multiple_positives/mAP@10: {mAP_multi:.4f}\n{'='*30}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print(f"Metadata not found at {metadata_path}, skipping multi-positive mAP.")       

if __name__ == "__main__":
    BI_CKPT = "checkpoints/cosmic-paper-15/epoch=13.ckpt"
    TRANS_CKPT = "checkpoints/revived-snowball-16/epoch=19.ckpt"
    
    # 第一步：训练门控网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_bi = AudioRetrievalModel.load_from_checkpoint(BI_CKPT).to(device)
    m_tr = AudioRetrievalModel.load_from_checkpoint(TRANS_CKPT).to(device)
    
    # 训练并获取门控模型
    # my_gate = train_gate(m_bi, m_tr) 
    
    # 第二步：使用门控进行推理
    run_gated_eval(BI_CKPT, TRANS_CKPT)