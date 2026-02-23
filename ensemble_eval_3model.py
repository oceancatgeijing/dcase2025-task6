import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import sys
import types
from dataclasses import dataclass

# --- 仅保留 aac-datasets 路径补丁 ---
@dataclass
class FakeAudioMetaData:
    sample_rate: int; num_frames: int; num_channels: int; bits_per_sample: int; encoding: str

backend = types.ModuleType("torchaudio.backend")
common = types.ModuleType("torchaudio.backend.common")
common.AudioMetaData = FakeAudioMetaData
backend.common = common
sys.modules["torchaudio.backend"] = backend
sys.modules["torchaudio.backend.common"] = common
# ----------------------------------

from d25_t6.retrieval_module import AudioRetrievalModel
from d25_t6.datasets.audio_loading import custom_loading
from aac_datasets import Clotho
from d25_t6.datasets.batch_collate import CustomCollate

def _batch_to_device(batch, device):
    """将 batch 中的张量移到指定设备（与 Lightning Trainer 行为一致）"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def min_max_normalize(matrix):
    """将分数矩阵归一化到 [0, 1] 之间"""
    m_min = matrix.min()
    m_max = matrix.max()
    return (matrix - m_min) / (m_max - m_min + 1e-8)

def run_ensemble_eval(bi_ckpt_path, trans_ckpt_path, third_ckpt_path=None, data_path='data', weights=None):
    """
    三模型集成评估
    
    Args:
        bi_ckpt_path: Bi-Encoder 模型路径
        trans_ckpt_path: Transformer 模型路径  
        third_ckpt_path: 第三个模型路径（如时序Transformer），可选
        data_path: 数据路径
        weights: 三模型权重列表，如 [0.33, 0.33, 0.34]，默认等权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置权重
    if weights is None:
        if third_ckpt_path:
            weights = [0.35, 0.4, 0.25]
            print("Using weights: 0.35, 0.4, 0.25")
        else:
            weights = [0.5, 0.5]
            print("Using equal weights: 0.5, 0.5")
    
    # 1. 加载模型
    print("Loading models...")
    model_bi = AudioRetrievalModel.load_from_checkpoint(bi_ckpt_path).to(device)
    model_trans = AudioRetrievalModel.load_from_checkpoint(trans_ckpt_path).to(device)
    
    model_bi.eval()
    model_trans.eval()
    
    # 加载第三个模型（如果提供）
    model_third = None
    if third_ckpt_path:
        print(f"Loading third model from: {third_ckpt_path}")
        model_third = AudioRetrievalModel.load_from_checkpoint(third_ckpt_path).to(device)
        model_third.eval()
        print(f"Third model config: temporal_aware={model_third.temporal_aware}, "
              f"training_mode={model_third.training_mode}")

    # 2. 准备测试数据 (使用 Clotho eval set)
    test_ds = custom_loading(Clotho(subset="eval", root=data_path, flat_captions=True))
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=CustomCollate()
    )

    all_audio_embeds_bi = []
    all_audio_embeds_trans = []
    all_audio_embeds_third = []  # 新增
    all_text_embeds_bi = []
    all_text_embeds_trans = []
    all_text_embeds_third = []   # 新增
    all_fnames = []
    all_captions = []

    # 3. 提取特征
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = _batch_to_device(batch, device)
            
            # Bi-Encoder 特征 (带 L2 归一化)
            a_bi = model_bi.forward_audio(batch)
            t_bi = model_bi.forward_text(batch)
            
            # Transformer 特征 (不带 L2 归一化，用于后续 Cross-Attention)
            a_trans, t_trans = model_trans.forward(batch)
            
            all_audio_embeds_bi.append(a_bi.cpu())
            all_text_embeds_bi.append(t_bi.cpu())
            all_audio_embeds_trans.append(a_trans.cpu())
            all_text_embeds_trans.append(t_trans.cpu())
            
            # 第三个模型特征提取
            if model_third:
                a_third, t_third = model_third.forward(batch)
                all_audio_embeds_third.append(a_third.cpu())
                all_text_embeds_third.append(t_third.cpu())
            
            all_fnames.extend(batch['fname'])
            all_captions.extend([c[0] for c in batch['captions']])

    # 合并并去重音频 (Clotho 每个音频对应 5 个 caption)
    audio_bi = torch.cat(all_audio_embeds_bi)
    audio_trans = torch.cat(all_audio_embeds_trans)
    text_bi = torch.cat(all_text_embeds_bi)
    text_trans = torch.cat(all_text_embeds_trans)
    
    # 第三个模型
    audio_third = None
    text_third = None
    if model_third:
        audio_third = torch.cat(all_audio_embeds_third)
        text_third = torch.cat(all_text_embeds_third)

    unique_paths = []
    select_indices = []
    targets = []
    path_to_idx = {}
    
    for i, p in enumerate(all_fnames):
        if p not in path_to_idx:
            path_to_idx[p] = len(unique_paths)
            unique_paths.append(p)
            select_indices.append(i)
        targets.append(path_to_idx[p])

    audio_bi = audio_bi[select_indices].to(device)
    audio_trans = audio_trans[select_indices].to(device)
    text_bi = text_bi.to(device)
    text_trans = text_trans.to(device)
    
    if model_third:
        audio_third = audio_third[select_indices].to(device)
        text_third = text_third.to(device)
        
    targets = np.array(targets)

    # 4. 计算分数矩阵
    print("Computing similarity matrices...")
    with torch.no_grad():
        # Bi-Encoder 分数 (Cosine Similarity)
        scores_bi = torch.matmul(text_bi, audio_bi.T).cpu().numpy()
        print(f"Bi-Encoder scores range: [{scores_bi.min():.2f}, {scores_bi.max():.2f}]")
        
        # Transformer 分数 (Cross-Attention Score)
        num_texts = text_trans.shape[0]
        num_audios = audio_trans.shape[0]
        scores_trans = np.zeros((num_texts, num_audios))
        
        for i in tqdm(range(num_texts), desc="Transformer Scoring"):
            s = model_trans.forward_fusion(audio_trans, text_trans[i].expand(num_audios, -1))
            scores_trans[i] = s.cpu().numpy()
        print(f"Transformer scores range: [{scores_trans.min():.2f}, {scores_trans.max():.2f}]")
        
        # 第三个模型分数
        scores_third = None
        if model_third:
            scores_third = np.zeros((num_texts, num_audios))
            
            # 判断第三个模型是否是时序感知模型
            is_temporal = hasattr(model_third, 'temporal_aware') and model_third.temporal_aware
            
            desc = "Third Model (Temporal) Scoring" if is_temporal else "Third Model Scoring"
            for i in tqdm(range(num_texts), desc=desc):
                if is_temporal:
                    # 时序Transformer使用特殊的融合函数
                    s = model_third.forward_fusion_temporal(
                        audio_third, 
                        text_third[i].expand(num_audios, -1)
                    )
                else:
                    # 标准Transformer融合
                    s = model_third.forward_fusion(
                        audio_third, 
                        text_third[i].expand(num_audios, -1)
                    )
                scores_third[i] = s.cpu().numpy()
            print(f"Third model scores range: [{scores_third.min():.2f}, {scores_third.max():.2f}]")

    # 5. 分数融合
    print("Fusing scores...")
    s_bi_norm = min_max_normalize(scores_bi)
    s_trans_norm = min_max_normalize(scores_trans)
    
    if model_third and scores_third is not None:
        s_third_norm = min_max_normalize(scores_third)
        # 三模型加权融合
        final_scores = (weights[0] * s_bi_norm + 
                       weights[1] * s_trans_norm + 
                       weights[2] * s_third_norm)
        print(f"Ensemble weights: Bi={weights[0]:.2f}, Trans={weights[1]:.2f}, Third={weights[2]:.2f}")
    else:
        # 双模型融合（原有逻辑）
        final_scores = weights[0] * s_bi_norm + weights[1] * s_trans_norm

    # 6. 计算指标
    print("Calculating metrics...")
    top_10 = np.argsort(-final_scores, axis=1)[:, :10]
    
    r1 = (top_10[:, :1] == targets[:, None]).any(axis=1).mean()
    r5 = (top_10[:, :5] == targets[:, None]).any(axis=1).mean()
    r10 = (top_10[:, :10] == targets[:, None]).any(axis=1).mean()
    
    # mAP
    aps = []
    for i in range(len(targets)):
        rank = np.where(top_10[i] == targets[i])[0]
        if len(rank) > 0:
            aps.append(1.0 / (rank[0] + 1))
        else:
            aps.append(0)
    mAP = np.mean(aps)

    print(f"\n--- Ensemble Results ---")
    print(f"R@1:  {r1:.4f}")
    print(f"R@5:  {r5:.4f}")
    print(f"R@10: {r10:.4f}")
    print(f"mAP@10: {mAP:.4f}")
    
    # 单独打印各模型指标（用于对比）
    print("\n--- Individual Model Performance (for reference) ---")
    for name, scores in [("Bi-Encoder", scores_bi), 
                         ("Transformer", scores_trans),
                         ("Third Model", scores_third if scores_third is not None else None)]:
        if scores is not None:
            top = np.argsort(-scores, axis=1)[:, :10]
            r1_i = (top[:, :1] == targets[:, None]).any(axis=1).mean()
            r5_i = (top[:, :5] == targets[:, None]).any(axis=1).mean()
            map_i = np.mean([1.0/(np.where(top[i]==targets[i])[0][0]+1) 
                           if len(np.where(top[i]==targets[i])[0])>0 else 0 
                           for i in range(len(targets))])
            print(f"{name}: R@1={r1_i:.4f}, R@5={r5_i:.4f}, mAP@10={map_i:.4f}")
    
    # test_multiple_positives/mAP@10（需 resources/metadata_eval.csv）
    metadata_path = "resources/metadata_eval.csv"
    if os.path.exists(metadata_path):
        matched_files = pd.read_csv(metadata_path)
        matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        captions_list = all_captions  # 与 final_scores 行顺序一致
        paths_list = unique_paths    # 与 final_scores 列顺序一致

        def get_ranks_np(scores_row, relevant_indices):
            """scores_row: (num_audios,), relevant_indices: list of int. 返回各相关音频的 0-based 排名"""
            r = np.asarray(relevant_indices)
            ranks = np.argsort(np.argsort(-np.asarray(scores_row)))[r]
            return ranks.tolist()

        matched_files["query_index"] = matched_files["query"].transform(
            lambda x: captions_list.index(x)
        )
        matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(
            lambda x: [paths_list.index(y) for y in x]
        )
        matched_files["TP_ranks"] = matched_files.apply(
            lambda row: get_ranks_np(final_scores[row["query_index"]], row["new_audio_indices"]),
            axis=1,
        )

        def average_precision_at_k(relevant_ranks, k=10):
            relevant_ranks = sorted(relevant_ranks)
            ap = 0.0
            for i, rank in enumerate(relevant_ranks, start=1):
                if rank >= k:
                    break
                ap += i / (rank + 1)
            return ap / len(relevant_ranks) if relevant_ranks else 0.0

        mAP_multi = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
        print(f"\ntest_multiple_positives/mAP@10: {mAP_multi:.4f}")
    else:
        print(f"\ntest_multiple_positives/mAP@10: (skip, no {metadata_path})")

if __name__ == "__main__":
    # 请替换为你真实的 ckpt 路径
    BI_CKPT = "checkpoints/cosmic-paper-15/epoch=13.ckpt"
    TRANS_CKPT = "checkpoints/revived-snowball-16/epoch=19.ckpt"
    
    # 新增的第三个模型（时序Transformer）
    THIRD_CKPT = "/root/autodl-tmp/dcase2025/checkpoints/firm-aardvark-26/epoch=17.ckpt"
    
    # 可选：自定义权重（默认等权重 1/3, 1/3, 1/3）
    # 如果第三个模型表现更好，可以给更高权重，如 [0.3, 0.3, 0.4]
    WEIGHTS = None  # 或 [0.33, 0.33, 0.34]
    
    run_ensemble_eval(BI_CKPT, TRANS_CKPT, THIRD_CKPT, weights=WEIGHTS)
    
    # 如果想测试双模型（不加第三个），把 THIRD_CKPT 设为 None：
    # run_ensemble_eval(BI_CKPT, TRANS_CKPT, None)