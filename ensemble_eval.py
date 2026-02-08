import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

from d25_t6.retrieval_module import AudioRetrievalModel
from d25_t6.datasets.audio_loading import custom_loading
from aac_datasets import Clotho
from d25_t6.datasets.batch_collate import CustomCollate

def min_max_normalize(matrix):
    """将分数矩阵归一化到 [0, 1] 之间"""
    m_min = matrix.min()
    m_max = matrix.max()
    return (matrix - m_min) / (m_max - m_min + 1e-8)

def run_ensemble_eval(bi_ckpt_path, trans_ckpt_path, data_path='data'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载两个模型
    print("Loading models...")
    model_bi = AudioRetrievalModel.load_from_checkpoint(bi_ckpt_path).to(device)
    model_trans = AudioRetrievalModel.load_from_checkpoint(trans_ckpt_path).to(device)
    
    model_bi.eval()
    model_trans.eval()

    # 2. 准备测试数据 (使用 Clotho eval set)
    test_ds = custom_loading(Clotho(subset="eval", root=data_path, flat_captions=True))
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=CustomCollate()
    )

    all_audio_embeds_bi = []
    all_audio_embeds_trans = []
    all_text_embeds_bi = []
    all_text_embeds_trans = []
    all_fnames = []
    all_captions = []

    # 3. 提取特征
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Bi-Encoder 特征 (带 L2 归一化)
            a_bi = model_bi.forward_audio(batch)
            t_bi = model_bi.forward_text(batch)
            
            # Transformer 特征 (不带 L2 归一化，用于后续 Cross-Attention)
            a_trans, t_trans = model_trans.forward(batch)
            
            all_audio_embeds_bi.append(a_bi.cpu())
            all_text_embeds_bi.append(t_bi.cpu())
            all_audio_embeds_trans.append(a_trans.cpu())
            all_text_embeds_trans.append(t_trans.cpu())
            all_fnames.extend(batch['fname'])
            all_captions.extend([c[0] for c in batch['captions']])

    # 合并并去重音频 (Clotho 每个音频对应 5 个 caption)
    audio_bi = torch.cat(all_audio_embeds_bi)
    audio_trans = torch.cat(all_audio_embeds_trans)
    text_bi = torch.cat(all_text_embeds_bi)
    text_trans = torch.cat(all_text_embeds_trans)

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
    targets = np.array(targets)

    # 4. 计算分数矩阵
    print("Computing similarity matrices...")
    with torch.no_grad():
        # Bi-Encoder 分数 (Cosine Similarity)
        scores_bi = torch.matmul(text_bi, audio_bi.T).cpu().numpy()
        
        # Transformer 分数 (Cross-Attention Score)
        num_texts = text_trans.shape[0]
        num_audios = audio_trans.shape[0]
        scores_trans = np.zeros((num_texts, num_audios))
        
        for i in tqdm(range(num_texts), desc="Transformer Scoring"):
            # 每次计算一个文本对所有音频的融合分
            s = model_trans.forward_fusion(audio_trans, text_trans[i].expand(num_audios, -1))
            scores_trans[i] = s.cpu().numpy()

    # 5. 分数融合
    print("Fusing scores...")
    s_bi_norm = min_max_normalize(scores_bi)
    s_trans_norm = min_max_normalize(scores_trans)
    
    # 综合分 = (Bi分 + Trans分) / 2
    final_scores = (s_bi_norm + s_trans_norm) / 2.0

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

if __name__ == "__main__":
    # 请替换为你真实的 ckpt 路径
    BI_CKPT = "checkpoints/spring-pine-8/epoch=19.ckpt"
    TRANS_CKPT = "checkpoints/transformer_exp/epoch=15.ckpt"
    
    run_ensemble_eval(BI_CKPT, TRANS_CKPT)