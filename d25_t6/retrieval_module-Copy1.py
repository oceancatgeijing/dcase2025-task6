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
        
        # 时序模式配置
        self.max_temporal_segments = kwargs.get('max_temporal_segments', 300)
        self.segment_duration = 10  # 每段10秒
        
        print(f"[模型配置] 时序感知: {self.temporal_aware}, "
              f"最大时段数: {self.max_temporal_segments}, 模式: {self.training_mode}")

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
            
            if self.temporal_aware:
                # ========== 修改：时间编码嵌入到音频段 ==========
                # 可学习的时段位置编码（相对位置：第0段、第1段...）
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(self.max_temporal_segments, 1024) * 0.02
                )
                
                # 可选：绝对时间编码（将秒数如0,10,20...编码为向量）
                # 使用正弦位置编码或线性层
                self.absolute_time_encoding = nn.Sequential(
                    nn.Linear(1, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024)
                )

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
        """标准融合（非时序模式）"""
        B = audio_embeds.shape[0]
        
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds.unsqueeze(1)
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(1)

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
        先时序注意力池化，再标准双向Cross-Attention
        audio_temporal: [B, T, 1024]（带时间编码的音频时段）
        text_embeds: [B, 1024] 或 [B, L, 1024]（纯文本特征）
        """
        B, T, D = audio_temporal.shape
    
        # 类型对齐
        target_dtype = self.audio_ln.weight.dtype
        if audio_temporal.dtype != target_dtype:
            audio_temporal = audio_temporal.to(target_dtype)
        if text_embeds.dtype != target_dtype:
            text_embeds = text_embeds.to(target_dtype)
        
        # 确保文本有序列维度 [B, 1, 1024]
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(1)
        
        # ========== 步骤1: 文本指导的时序注意力池化 ==========
        # 使用第一层计算文本对音频的注意力权重，并融合音频
        with torch.no_grad():  # 仅作为池化权重，不计算梯度（避免与步骤2重复）
            _, attn_weights = self.text_to_audio_attn[0](
                text_embeds,      # Query: [B, 1, 1024]
                audio_temporal,   # Key: [B, T, 1024]
                audio_temporal    # Value: [B, T, 1024]
            )
            # attn_weights: [B, 1, T]，表示文本对每个音频时段的关注度
        
        # 加权融合：将T个时段压缩为1个向量 [B, 1, 1024]
        audio_fused = torch.bmm(attn_weights, audio_temporal)
        
        # 对融合后的音频做LayerNorm
        audio_fused = self.audio_ln(audio_fused)
        
        # ========== 步骤2: 类似非时序模式的双向Cross-Attention ==========
        # 此时：text_embeds: [B, 1, 1024], audio_fused: [B, 1, 1024]
        # 类似于 forward_fusion 的处理逻辑
        
        text_feat = text_embeds      # [B, 1, 1024]
        audio_feat = audio_fused     # [B, 1, 1024]
        
        for i in range(self.num_fusion_layers):
            # 方向1: 文本查音频（融合后的单向量）
            text_ctx, _ = self.text_to_audio_attn[i](
                text_feat,      # [B, 1, 1024]
                audio_feat,     # [B, 1, 1024]
                audio_feat      # [B, 1, 1024]
            )
            text_feat = self.text_ln(text_feat + text_ctx)
            
            # 方向2: 音频查文本
            audio_ctx, _ = self.audio_to_text_attn[i](
                audio_feat,     # [B, 1, 1024]
                text_feat,      # [B, 1, 1024]（使用更新后的文本）
                text_feat       # [B, 1, 1024]
            )
            audio_feat = self.audio_ln(audio_feat + audio_ctx)
        
        # 最终聚合：拼接交互后的两者
        context = torch.cat([text_feat, audio_feat], dim=1)  # [B, 2, 1024]
        
        # CLS token聚合
        x = self.cls_token.expand(B, -1, -1)  # [B, 1, 1024]
        fusion_feature, _ = self.output_attn(query=x, key=context, value=context)
        score = self.score_head(fusion_feature.squeeze(1))
        return score.squeeze(-1)
        
    def aggregate_audio_segments(self, audio_embeddings, durations):
        """非时序聚合"""
        batch_size = audio_embeddings.shape[0]
        aggregated = []
        for i, duration in enumerate(durations):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))
        aggregated = torch.stack(aggregated)
        aggregated = self.audio_projection(aggregated)
        return aggregated
    
    def forward(self, batch):
        # 获取音频特征 [B, N_seg, 768]
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))
        
        if self.training_mode == 'transformer':
            if self.temporal_aware:
                # ========== 音频侧：时序编码 ==========
                B, N_actual, D = audio_embeddings.shape
                
                if N_actual > self.max_temporal_segments:
                    if self.training:
                        print(f"警告: 音频分段数{N_actual}超过最大支持{self.max_temporal_segments}，已截断")
                    N_actual = self.max_temporal_segments
                    audio_embeddings = audio_embeddings[:, :N_actual, :]
                
                # 投影到1024维
                audio_segments = []
                for i in range(N_actual):
                    seg_proj = self.audio_projection(audio_embeddings[:, i, :])
                    audio_segments.append(seg_proj)
                a_feat = torch.stack(audio_segments, dim=1)  # [B, N_actual, 1024]
                
                # ========== 关键：时间编码嵌入到音频段 ==========
                # 1. 可学习的位置编码（第0段、第1段...）
                a_feat = a_feat + self.temporal_pos_embedding[:N_actual].unsqueeze(0)
                
                # 2. 可选：绝对时间编码（将实际秒数编码为向量）
                # 生成时间戳张量 [0, 10, 20, ...]
                time_stamps = torch.arange(
                    0, N_actual * self.segment_duration, self.segment_duration,
                    device=a_feat.device, dtype=torch.float32
                ).view(1, N_actual, 1)  # [1, N_actual, 1]
                
                # 将秒数编码为1024维向量并添加
                time_encoding = self.absolute_time_encoding(time_stamps)  # [1, N_actual, 1024]
                a_feat = a_feat + time_encoding
                
                # 不进行L2归一化，保留幅度
                
            else:
                # 非时序
                a_feat = self.aggregate_audio_segments(audio_embeddings, batch['duration'])
                a_feat = self.audio_ln(a_feat)
            
            # ========== 文本侧：标准处理（无时间戳前缀）==========
            # 移除了时间戳前缀生成，恢复标准文本清洗
            captions = []
            for i, b in enumerate([c[0] for c in batch['captions']]):
                if not isinstance(b, str): 
                    b = b[0]
                captions.append(b.lower().translate(str.maketrans('', '', string.punctuation)))
            
            # 标准长度32（无需为时间戳预留空间）
            tokenized = self.tokenizer(
                captions, 
                add_special_tokens=True, 
                padding='max_length', 
                return_tensors='pt', 
                max_length=32,  # 恢复标准长度
                truncation=True
            ).to(self.device)
    
            t_feat = self.text_embedding_model(
                input_ids=tokenized['input_ids'], 
                attention_mask=tokenized['attention_mask']
            )[0][:, 0, :]  # [B, 1024]
            t_feat = self.text_projection(t_feat)
            
            # 时序模式下不做LayerNorm（保留幅度给attention）
            if not self.temporal_aware:
                t_feat = self.text_ln(t_feat)
    
            return a_feat, t_feat
        else:
            # Bi-encoder
            audio_emb = self.forward_audio(batch)
            text_emb = self.forward_text(batch)
            return audio_emb, text_emb

    def forward_audio(self, batch):
        """Bi-encoder音频编码"""
        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1))
        
        if self.temporal_aware:
            B, T, D = audio_embeddings.shape
            if T > self.max_temporal_segments:
                T = self.max_temporal_segments
                audio_embeddings = audio_embeddings[:, :T, :]
            
            audio_temp = []
            for i in range(T):
                seg = self.audio_projection(audio_embeddings[:, i, :])
                audio_temp.append(seg)
            audio_emb = torch.stack(audio_temp, dim=1).mean(dim=1)
        else:
            audio_emb = self.aggregate_audio_segments(audio_embeddings, batch['duration'])
            
        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        return audio_emb

    def forward_text(self, batch):
        """Bi-encoder文本编码（标准处理，无时间戳）"""
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
            a_feat, t_feat = self.forward(batch)
            C = torch.zeros((batch_size, batch_size), device=self.device)
            
            if self.temporal_aware:
                # 时序Transformer
                for i in range(batch_size):
                    audio_i = a_feat[i].unsqueeze(0).expand(batch_size, -1, -1)
                    C[i] = self.forward_fusion_temporal(audio_i, t_feat)
            else:
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
        with torch.no_grad():
            audio_embeddings, text_embeddings = self.forward(batch)
        args = {
            'audio_embeddings': audio_embeddings.detach().cpu(),
            'text_embeddings': text_embeddings.detach().cpu(),
            'caption': [c[0] for c in batch['captions']],
            'path': batch['fname']
        }
        self.validation_outputs.append(args)

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs
        if not outputs:
            return
            
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

        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select].to(self.device)
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs]).to(self.device)

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
                        C[i] = self.forward_fusion_temporal(
                            audio_embeddings, 
                            text_embeddings[i].expand(num_audios, -1)
                        )
                    else:
                        C[i] = self.forward_fusion(
                            audio_embeddings, 
                            text_embeddings[i].expand(num_audios, -1)
                        )
        
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
                num_hits = 0
                for i, rank in enumerate(relevant_ranks, start=1):
                    if rank < k:
                        num_hits += 1
                        ap += num_hits / (rank + 1)
                return ap / len(relevant_ranks) if relevant_ranks else 0.0

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