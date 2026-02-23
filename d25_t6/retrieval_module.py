import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel

from d25_t6.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper


class AudioRetrievalModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.training_mode = kwargs.get('training_mode', 'bi-encoder')
        self.temporal_aware = kwargs.get('temporal_aware', False)
        print(f"[模型配置] 时序感知模式: {self.temporal_aware}, 训练模式: {self.training_mode}")

        # audio encoder
        self.audio_embedding_model = CutInputIntoSegmentsWrapper(
            PaSSTSNoOverlapWrapper(
                s_patchout_t=kwargs['s_patchout_t'],
                s_patchout_f=kwargs['s_patchout_f']
            ),
            max_input_length=10*32000,
            segment_length=10*32000,
            hop_size=10*32000
        )
        
        # 投影层：768 -> 1024
        self.audio_projection = torch.nn.Linear(768, 1024)

        # text encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.text_embedding_model = RobertaModel.from_pretrained(
            'roberta-base' if kwargs['roberta_base'] else 'roberta-large',
            add_pooling_layer=False,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            output_hidden_states=False
        )
        self.text_projection = torch.nn.Linear(768 if kwargs['roberta_base'] else 1024, 1024)

        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])
        
        if self.training_mode == 'transformer':
            self.num_fusion_layers = 6
            self.audio_to_text_attn = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                for _ in range(self.num_fusion_layers)
            ])
            self.text_to_audio_attn = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
                for _ in range(self.num_fusion_layers)
            ])
            
            self.cls_token = torch.nn.Parameter(torch.empty(1, 1, 1024))
            torch.nn.init.normal_(self.cls_token, std=0.02)
            
            self.output_attn = torch.nn.MultiheadAttention(
                embed_dim=1024, 
                num_heads=8, 
                batch_first=True
            )
            
            self.score_head = torch.nn.Linear(1024, 1)
            self.audio_ln = torch.nn.LayerNorm(1024)
            self.text_ln = torch.nn.LayerNorm(1024)
            
            # 时序感知的额外投影（可选：用于将文本映射到查询空间）
            if self.temporal_aware:
                self.temporal_pos_embedding = nn.Parameter(torch.randn(3, 1024) * 0.02)
                # 文本 -> 查询向量，用于查询音频时段
                self.temporal_query_proj = torch.nn.Linear(1024, 1024)

        self.validation_outputs = []
        self.kwargs = kwargs
        self.compile_model()
    
    def compile_model(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward_fusion(self, audio_embeds, text_embeds):
        """
        标准融合：适用于非时序模式 [B, D]
        audio_embeds: [B, 1024] 
        text_embeds: [B, 1024]
        """
        B = audio_embeds.shape[0]
        
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds.unsqueeze(1)  # [B, 1, 1024]
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(1)    # [B, 1, 1024]

        x = self.cls_token.expand(B, -1, -1)

        for i in range(self.num_fusion_layers):
            text_ctx, _ = self.text_to_audio_attn[i](text_embeds, audio_embeds, audio_embeds)
            text_embeds = self.text_ln(text_embeds + text_ctx)
            audio_ctx, _ = self.audio_to_text_attn[i](audio_embeds, text_embeds, text_embeds)
            audio_embeds = self.audio_ln(audio_embeds + audio_ctx)

        context = torch.cat([audio_embeds, text_embeds], dim=1)
        fusion_feature, _ = self.output_attn(query=x, key=context, value=context)
        score = self.score_head(fusion_feature.squeeze(1))
        return score.squeeze(-1)
    
    def forward_fusion_temporal(self, audio_temporal, text_embeds):
        """
        细粒度时序对齐融合：
        audio_temporal: [B, T, 1024] (T=3个时段)
        text_embeds: [B, 1024] 或 [B, L, 1024]
        实现：文本作为Query，关注音频时段（Key/Value）
        """
        B, T, D = audio_temporal.shape

        # ========== 新增：类型对齐（适配混合精度训练/推理）==========
        target_dtype = self.audio_ln.weight.dtype
        if audio_temporal.dtype != target_dtype:
            audio_temporal = audio_temporal.to(target_dtype)
        if text_embeds.dtype != target_dtype:
            text_embeds = text_embeds.to(target_dtype)
        # ============================================================
        
        # 确保文本有序列维度 [B, L, D]
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(1)  # [B, 1, 1024]
            L = 1
        else:
            L = text_embeds.shape[1]
        
        # 时序感知交叉注意力
        # 策略：让文本去关注音频时段（细粒度对齐）
        audio_temporal = self.audio_ln(audio_temporal)  # Pre-norm
        
        # 逐层交互：文本 Query -> 音频 Key/Value
        text_query = text_embeds
        for i in range(self.num_fusion_layers):
            # 文本关注音频时段 [B, L, D] x [B, T, D] -> [B, L, D]
            text_ctx, attn_weights = self.text_to_audio_attn[i](
                text_query,           # Query: 文本
                audio_temporal,       # Key: 音频时段
                audio_temporal        # Value: 音频时段
            )
            text_query = self.text_ln(text_query + text_ctx)
            
            # 可选：音频也关注文本（双向）
            # audio_ctx, _ = self.audio_to_text_attn[i](audio_temporal, text_query, text_query)
            # audio_temporal = self.audio_ln(audio_temporal + audio_ctx)
        
        # 聚合：使用注意力权重对音频进行加权
        # attn_weights: [B, L, T]，表示每个文本词对每个音频时段的注意力
        
        # 方法1：取最后一个层的注意力权重，对音频做加权平均
        # weighted_audio = torch.bmm(attn_weights, audio_temporal)  # [B, L, D]
        
        # 方法2：使用CLS token提取融合信息（类似原逻辑）
        # 这里我们使用 text_query（已经融合了音频信息）与原始文本结合
        context = torch.cat([text_query, audio_temporal], dim=1)  # [B, L+T, D]
        
        # CLS token 查询所有信息
        x = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        fusion_feature, _ = self.output_attn(query=x, key=context, value=context)
        
        score = self.score_head(fusion_feature.squeeze(1))  # [B, 1]
        
        # 可选：额外监督注意力分布（如稀疏性约束）
        # 可以在这里返回 attn_weights 用于辅助 loss
        
        return score.squeeze(-1)
        
    def aggregate_audio_segments(self, audio_embeddings, durations):
        """
        非时序模式：压缩时序维度
        audio_embeddings: [B, N_seg, 768]
        """
        batch_size = audio_embeddings.shape[0]
        aggregated = []
        for i, duration in enumerate(durations):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))
        aggregated = torch.stack(aggregated)  # [B, 768]
        aggregated = self.audio_projection(aggregated)  # [B, 1024]
        return aggregated
    
    def forward(self, batch):
        # 获取原始分段特征 [B, N_seg, 768]
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))
        
        if self.training_mode == 'transformer':
            # ========== 1. 音频特征处理（时序/非时序分支）==========
            if self.temporal_aware:
                # 时序模式：保持 [B, T, 1024]，逐段投影
                B, T, D = audio_embeddings.shape
                audio_temp = []
                for i in range(T):
                    seg = self.audio_projection(audio_embeddings[:, i, :])  # [B, 1024]
                    audio_temp.append(seg)
                a_feat = torch.stack(audio_temp, dim=1)  # [B, T, 1024]
                
                # 添加可学习的时序位置编码 [B, 3, 1024]
                a_feat = a_feat + self.temporal_pos_embedding.unsqueeze(0)
                
                # 不进行L2归一化，保留幅度给注意力机制
            else:
                # 非时序：压缩为 [B, 1024]
                a_feat = self.aggregate_audio_segments(audio_embeddings, batch['duration'])
                a_feat = self.audio_ln(a_feat)  # LayerNorm
            
            # ========== 2. 文本特征处理（带时序标记）==========
            if self.temporal_aware:
                # 时序模式：添加时间戳标记，增强时序对齐
                enhanced_captions = []
                for i, b in enumerate([c[0] for c in batch['captions']]):
                    if not isinstance(b, str): 
                        b = b[0]
                    base_caption = b.lower().translate(str.maketrans('', '', string.punctuation))
                    
                    # 方案A：前缀时间戳（推荐，计算高效）
                    marked_caption = f"[0-10s] [10-20s] [20-30s] {base_caption}"
                    
                    # 方案B：分段重复描述（时序信号更强，但序列更长）
                    # marked_caption = f"[0-10s] {base_caption} [10-20s] {base_caption} [20-30s] {base_caption}"
                    
                    enhanced_captions.append(marked_caption)
                
                captions = enhanced_captions
                max_length = 48  # 增加长度容纳时间戳 [0-10s]等
            else:
                # 非时序模式：保持原有处理
                captions = []
                for i, b in enumerate([c[0] for c in batch['captions']]):
                    if not isinstance(b, str): 
                        b = b[0]
                    captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))
                max_length = 32
                
            tokenized = self.tokenizer(
                captions, 
                add_special_tokens=True, 
                padding='max_length', 
                return_tensors='pt', 
                max_length=max_length,  # 动态长度
                truncation=True
            ).to(self.device)
    
            t_feat = self.text_embedding_model(
                input_ids=tokenized['input_ids'], 
                attention_mask=tokenized['attention_mask']
            )[0][:, 0, :]  # [B, 1024]
            t_feat = self.text_projection(t_feat)
            
            # 时序模式下不对文本做LayerNorm（保留幅度给attention），非时序下做norm
            if not self.temporal_aware:
                t_feat = self.text_ln(t_feat)
    
            return a_feat, t_feat
        else:
            # Bi-encoder 模式（保持原逻辑不变）
            audio_emb = self.forward_audio(batch)
            text_emb = self.forward_text(batch)
            return audio_emb, text_emb

    def forward_audio(self, batch):
        """Bi-encoder 音频编码"""
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))
        
        if self.temporal_aware:
            # 时序模式也压缩（Bi-encoder下需要全局向量计算相似度）
            B, T, D = audio_embeddings.shape
            audio_temp = []
            for i in range(T):
                seg = self.audio_projection(audio_embeddings[:, i, :])
                audio_temp.append(seg)
            audio_emb = torch.stack(audio_temp, dim=1).mean(dim=1)  # 平均池化
        else:
            audio_emb = self.aggregate_audio_segments(audio_embeddings, batch['duration'])
            
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        return audio_emb

    def forward_text(self, batch):
        """文本编码（保持原逻辑）"""
        device = self.device
        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not isinstance(b, str):
                b = b[0]
            captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))

        tokenized = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        sentence_features = token_embeddings[:, 0, :]
        sentence_features = self.text_projection(sentence_features)
        sentence_features = F.normalize(sentence_features, p=2, dim=-1)
        return sentence_features

    def training_step(self, batch, batch_idx):
        self.lr_scheduler_step(batch_idx)

        paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
        I = torch.tensor(paths[None, :] == paths[:, None], device=self.device)
        batch_size = len(paths)

        if self.training_mode == 'bi-encoder':
            audio_embeddings, text_embeddings = self.forward(batch)
            C = torch.matmul(audio_embeddings, text_embeddings.T) / torch.abs(self.tau)
        else:
            a_feat, t_feat = self.forward(batch)  # a_feat: [B, T, 1024] 或 [B, 1024]
            
            C = torch.zeros((batch_size, batch_size), device=self.device)
            
            if self.temporal_aware:
                # 时序Transformer：使用细粒度对齐
                for i in range(batch_size):
                    # 第i个音频（时序）与所有文本的融合分数
                    # a_feat[i]: [T, 1024]，需要扩展为 [B, T, 1024]
                    audio_i = a_feat[i].unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, 1024]
                    C[i] = self.forward_fusion_temporal(audio_i, t_feat)
            else:
                # 标准Transformer
                for i in range(batch_size):
                    C[i] = self.forward_fusion(a_feat[i].expand(batch_size, -1), t_feat)

        C_audio = torch.log_softmax(C, dim=0) 
        C_text = torch.log_softmax(C, dim=1)
        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())

        self.log("train/loss", loss, batch_size=batch_size, prog_bar=True, sync_dist=True)
        if self.training_mode == 'bi-encoder':
            self.log('train/tau', torch.abs(self.tau), sync_dist=True)
        self.log('train/temporal_aware', float(self.temporal_aware), sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        audio_embeddings, text_embeddings = self.forward(batch)
        args = {
            'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
            'text_embeddings': copy.deepcopy(text_embeddings.detach()),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }
        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        target = []
        select = []
        first_occurrence = {}
        for i, p in enumerate(paths):
            index = first_occurrence.get(p)
            if index is None:
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i)
            target.append(index)
        paths = paths[select]

        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        if self.training_mode == 'bi-encoder':
            C = torch.matmul(text_embeddings, audio_embeddings.T)
        else:
            print(f"Transformer 模式评估中，时序感知: {self.temporal_aware}...")
            num_texts = text_embeddings.shape[0]
            num_audios = audio_embeddings.shape[0]
            C = torch.zeros((num_texts, num_audios), device=self.device)
            
            with torch.no_grad():
                for i in range(num_texts):
                    if self.temporal_aware:
                        # 时序模式：音频是 [num_audios, T, D]
                        C[i] = self.forward_fusion_temporal(audio_embeddings, text_embeddings[i].expand(num_audios, -1))
                    else:
                        C[i] = self.forward_fusion(audio_embeddings, text_embeddings[i].expand(num_audios, -1))
        
        top_ten = C.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        r_1 = (top_ten[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5 = (top_ten[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10 = (top_ten == target[:, None]).sum(axis=1).mean()

        AP = 1 / ((top_ten == target[:, None]).argmax(axis=1) + 1)
        AP[~(top_ten == target[:, None]).any(axis=1)] = 0
        mAP = AP.mean()

        self.log(f'{prefix}/R@1', r_1)
        self.log(f'{prefix}/R@5', r_5)
        self.log(f'{prefix}/R@10', r_10)
        self.log(f'{prefix}/mAP@10', mAP)

        if os.path.exists(f'resources/metadata_eval.csv') and prefix == 'test':
            matched_files = pd.read_csv(f'resources/metadata_eval.csv')
            matched_files["audio_filenames"] = matched_files["audio_filenames"].transform(lambda x: ast.literal_eval(x))

            def get_ranks(c, r):
                ranks = [i.item() for i in torch.argsort(torch.argsort(-c))[r]]
                return ranks

            matched_files["query_index"] = matched_files["query"].transform(lambda x: captions.tolist().index(x))
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(lambda x: [paths.tolist().index(y) for y in x])
            matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C[row["query_index"]], row["new_audio_indices"]), axis=1)

            def average_precision_at_k(relevant_ranks, k=10):
                relevant_ranks = sorted(relevant_ranks)
                ap = 0.0
                for i, rank in enumerate(relevant_ranks, start=1):
                    if rank >= k:
                        break
                    ap += i / (rank + 1)
                return ap / len(relevant_ranks)

            new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
            self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)
            
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False
        )
        return optimizer

    def lr_scheduler_step(self, batch_idx):
        steps_per_epoch = self.trainer.num_training_batches
        min_lr = self.kwargs['min_lr']
        max_lr = self.kwargs['max_lr']
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)