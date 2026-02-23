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
    """将 batch 中的张量移到指定设备"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def analyze_recall_quality(scores_bi, targets, K_values=[100, 500, 1000]):
    """
    分析Bi-Encoder的召回质量，检查级联瓶颈
    如果 R@K > 0.99，说明瓶颈在精排阶段
    如果 R@K < 0.90，说明需要增大K
    """
    print(f"\n{'='*50}")
    print("--- Recall Quality Analysis (Bi-Encoder) ---")
    print(f"{'='*50}")
    
    results = {}
    for K in K_values:
        top_k_indices = np.argsort(-scores_bi, axis=1)[:, :K]
        recall_at_k = np.mean([targets[i] in top_k_indices[i] for i in range(len(targets))])
        results[K] = recall_at_k
        print(f"Bi-Encoder R@{K}: {recall_at_k:.4f}")
    
    # 关键诊断建议
    max_recall = results[max(K_values)]
    if max_recall > 0.99:
        print("\n[诊断] 召回率接近100%，瓶颈在精排/融合阶段")
        print("建议：优化Transformer重排策略或改用特征级融合")
    elif max_recall < 0.90:
        print(f"\n[诊断] 即使R@{max(K_values)}也低于90%，召回不足")
        print("建议：增大K值，或检查Bi-Encoder单独性能")
    else:
        print(f"\n[诊断] 召回率良好({max_recall:.2%})，但仍有提升空间")
        print("建议：尝试自适应门控或增大当前K值")
    
    return results

def run_cascade_eval(bi_ckpt_path, trans_ckpt_path, data_path='data', K=500, alpha=0.3):
    """
    级联检索评估 (Cascade Retrieval)
    
    Args:
        K: Bi-Encoder召回的候选数量 (默认500，可根据显存调整)
        alpha: Bi-Encoder分数权重 (默认0.3)，Transformer权重为(1-alpha)
               如果alpha=0.0则纯依赖Transformer重排，alpha=1.0则退化为纯Bi-Encoder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nCascade Config: Bi-Encoder Recall Top-{K}, Fusion Weight α={alpha}")

    # 1. 加载两个模型
    print("Loading models...")
    model_bi = AudioRetrievalModel.load_from_checkpoint(bi_ckpt_path).to(device)
    model_trans = AudioRetrievalModel.load_from_checkpoint(trans_ckpt_path).to(device)
    
    model_bi.eval()
    model_trans.eval()

    # 2. 准备测试数据
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
            batch = _batch_to_device(batch, device)
            # Bi-Encoder 特征 (L2归一化)
            a_bi = model_bi.forward_audio(batch)
            t_bi = model_bi.forward_text(batch)
            
            # Transformer 特征 (用于Cross-Attention)
            a_trans, t_trans = model_trans.forward(batch)
            
            all_audio_embeds_bi.append(a_bi.cpu())
            all_text_embeds_bi.append(t_bi.cpu())
            all_audio_embeds_trans.append(a_trans.cpu())
            all_text_embeds_trans.append(t_trans.cpu())
            all_fnames.extend(batch['fname'])
            all_captions.extend([c[0] for c in batch['captions']])

    # 合并并去重音频
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
    
    num_texts = text_bi.shape[0]
    num_audios = audio_bi.shape[0]
    print(f"Texts: {num_texts}, Unique Audios: {num_audios}")

    # 4. 第一阶段：Bi-Encoder全库检索 (快速召回)
    print(f"\nStage 1: Bi-Encoder Full Retrieval...")
    with torch.no_grad():
        scores_bi = torch.matmul(text_bi, audio_bi.T).cpu().numpy()  # [N_text, N_audio]
    
    # ===== 新增：召回质量诊断 =====
    recall_stats = analyze_recall_quality(scores_bi, targets, K_values=[100, 500, 1000, num_audios])
    # ==============================
    
    # 5. 第二阶段：级联精排 (Cascade Reranking)
    print(f"\nStage 2: Transformer Reranking Top-{K} candidates...")
    
    # 初始化最终分数矩阵（以Bi-Encoder为基准，避免漏检）
    final_scores = scores_bi.copy()
    
    # 统计信息
    rerank_stats = {
        'trans_scores_mean': [],
        'bi_scores_mean': [],
        'improved_queries': 0,
        'degraded_queries': 0,
        'missed_in_topk': 0  # 新增：统计有多少正样本不在Top-K内
    }
    
    with torch.no_grad():
        for i in tqdm(range(num_texts), desc="Reranking"):
            # 召回Top-K候选
            k_actual = min(K, num_audios)
            top_k_scores_bi, top_k_indices = torch.topk(
                torch.tensor(scores_bi[i]), k=k_actual
            )
            top_k_indices = top_k_indices.cpu().numpy()
            
            # 检查正样本是否在Top-K内（用于分析）
            if targets[i] not in top_k_indices:
                rerank_stats['missed_in_topk'] += 1
            
            # 获取候选音频的Transformer特征 [K, D]
            candidate_audio = audio_trans[top_k_indices]
            
            # 当前文本特征扩展到K个
            text_expanded = text_trans[i].unsqueeze(0).expand(k_actual, -1)
            
            # Transformer重排序 (Cross-Attention计算)
            trans_scores = model_trans.forward_fusion(candidate_audio, text_expanded)
            trans_scores = trans_scores.cpu().numpy().flatten()
            
            # 归一化Transformer分数到[0,1]（与Bi-Encoder尺度对齐）
            if trans_scores.max() != trans_scores.min():
                trans_scores_norm = (trans_scores - trans_scores.min()) / (trans_scores.max() - trans_scores.min())
            else:
                trans_scores_norm = trans_scores
            
            # 归一化Bi-Encoder的Top-K分数
            bi_topk = scores_bi[i, top_k_indices]
            if bi_topk.max() != bi_topk.min():
                bi_topk_norm = (bi_topk - bi_topk.min()) / (bi_topk.max() - bi_topk.min())
            else:
                bi_topk_norm = bi_topk
            
            # 融合分数：加权平均（关键！避免纯Transformer决策导致的漏检）
            fused_scores = alpha * bi_topk_norm + (1 - alpha) * trans_scores_norm
            
            # 更新最终分数（仅更新Top-K位置，其余保持Bi-Encoder原值）
            final_scores[i, top_k_indices] = fused_scores
            
            # 统计：检查重排是否改善了Top-1（调试用）
            orig_rank = np.argsort(-scores_bi[i])
            new_rank = np.argsort(-final_scores[i])
            if targets[i] in top_k_indices:
                orig_pos = np.where(orig_rank == targets[i])[0][0]
                new_pos = np.where(new_rank == targets[i])[0][0]
                if new_pos < orig_pos:
                    rerank_stats['improved_queries'] += 1
                elif new_pos > orig_pos:
                    rerank_stats['degraded_queries'] += 1
            
            rerank_stats['trans_scores_mean'].append(trans_scores.mean())
            rerank_stats['bi_scores_mean'].append(bi_topk.mean())

    # 6. 计算指标
    print("\nCalculating metrics...")
    top_10 = np.argsort(-final_scores, axis=1)[:, :10]
    
    r1 = (top_10[:, :1] == targets[:, None]).any(axis=1).mean()
    r5 = (top_10[:, :5] == targets[:, None]).any(axis=1).mean()
    r10 = (top_10[:, :10] == targets[:, None]).any(axis=1).mean()
    
    # mAP@10
    aps = []
    for i in range(len(targets)):
        rank = np.where(top_10[i] == targets[i])[0]
        if len(rank) > 0:
            aps.append(1.0 / (rank[0] + 1))
        else:
            aps.append(0)
    mAP = np.mean(aps)

    # 单模型对比（用于分析级联效果）
    top_10_bi = np.argsort(-scores_bi, axis=1)[:, :10]
    r1_bi = (top_10_bi[:, :1] == targets[:, None]).any(axis=1).mean()
    
    print(f"\n{'='*50}")
    print(f"--- Cascade Retrieval Results (K={K}, α={alpha}) ---")
    print(f"{'='*50}")
    print(f"R@1:  {r1:.4f}  (Bi-Encoder alone: {r1_bi:.4f})")
    print(f"R@5:  {r5:.4f}")
    print(f"R@10: {r10:.4f}")
    print(f"mAP@10: {mAP:.4f}")
    print(f"\nRerank Analysis:")
    print(f"  - Improved queries: {rerank_stats['improved_queries']} ({rerank_stats['improved_queries']/num_texts*100:.1f}%)")
    print(f"  - Degraded queries: {rerank_stats['degraded_queries']} ({rerank_stats['degraded_queries']/num_texts*100:.1f}%)")
    print(f"  - Missed in Top-{K}: {rerank_stats['missed_in_topk']} ({rerank_stats['missed_in_topk']/num_texts*100:.1f}%) [不可被级联挽救]")
    
    # test_multiple_positives/mAP@10
    metadata_path = "resources/metadata_eval.csv"
    if os.path.exists(metadata_path):
        matched_files = pd.read_csv(metadata_path)
        matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        captions_list = all_captions
        paths_list = unique_paths

        def get_ranks_np(scores_row, relevant_indices):
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
        print(f"test_multiple_positives/mAP@10: {mAP_multi:.4f}")
    else:
        print(f"test_multiple_positives/mAP@10: (skip, no {metadata_path})")
    
    return {
        'r1': r1, 'r5': r5, 'r10': r10, 'mAP': mAP,
        'scores': final_scores, 'targets': targets,
        'recall_stats': recall_stats  # 返回召回统计供分析
    }

if __name__ == "__main__":
    BI_CKPT = "checkpoints/cosmic-paper-15/epoch=13.ckpt"
    TRANS_CKPT = "checkpoints/revived-snowball-16/epoch=19.ckpt"
    
    # 建议参数：
    # K=500: 召回500个候选（平衡速度与精度）
    # alpha=0.3: 保留30%Bi-Encoder分数，避免漏检（可尝试0.0/0.5/0.7对比）
    run_cascade_eval(BI_CKPT, TRANS_CKPT, K=500, alpha=0.7)