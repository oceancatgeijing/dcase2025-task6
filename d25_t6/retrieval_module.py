import copy
import math
import string
from typing import Any
import os
import ast
import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from transformers import RobertaTokenizer, RobertaModel

from d25_t6.passt import CutInputIntoSegmentsWrapper, PaSSTSNoOverlapWrapper


class AudioRetrievalModel(pl.LightningModule):

    def __init__(self, **kwargs):

        super().__init__()
        self.save_hyperparameters(kwargs)
        # 可选项: 'bi-encoder', 'transformer'，双编码器和transformer
        self.training_mode = kwargs.get('training_mode', 'bi-encoder') 
        

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
            # 定义一个简单的 Fusion Transformer
            # 这里使用标准 Transformer 层，输入维度为 1024
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
            self.fusion_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
            # 最终输出一个标量分数
            self.score_head = torch.nn.Linear(1024, 1)

        self.validation_outputs = []

        self.kwargs = kwargs

        self.compile_model()

    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()  # Get current GPU device
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward_fusion(self, audio_embeds, text_embeds):
            """
            Transformer 融合逻辑：将音频和文本特征序列化后融合
            """
            # 假设 audio_embeds: [B, 1024], text_embeds: [B, 1024]
            # 构造序列输入: [B, 2, 1024]
            fusion_input = torch.stack([audio_embeds, text_embeds], dim=1) 
            
            # 通过 Transformer 提取融合特征
            fusion_output = self.fusion_transformer(fusion_input) # [B, 2, 1024]
            
            # 取第一个 token 或平均值作为融合特征，并计算得分
            combined_feature = fusion_output.mean(dim=1) # [B, 1024]
            score = self.score_head(combined_feature) # [B, 1]
            return score.squeeze(-1)
    
    def forward(self, batch) -> Any:

        # embed audio & text
        text_embeddings = self.forward_text(batch)
        audio_embeddings = self.forward_audio(batch)

        return audio_embeddings, text_embeddings

    def forward_audio(self, batch):

        audio_embeddings = self.audio_embedding_model(batch['audio'].mean(1)) # forward

        # mask embeddings from padded empty audio parts
        aggregated = []
        for i, duration in enumerate(batch['duration']):
            if duration <= 10:
                aggregated.append(audio_embeddings[i, 0])
            elif duration <= 20:
                aggregated.append(audio_embeddings[i, :2].mean(-2))
            else:
                aggregated.append(audio_embeddings[i].mean(-2))

        audio_embeddings = torch.stack(aggregated)
        audio_embeddings = self.audio_projection(audio_embeddings) # project to same dimension
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1) # normalize
        return audio_embeddings

    def forward_text(self, batch):

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # 动态获取设备
        device = self.device

        captions = []
        for i, b in enumerate([c[0] for c in batch['captions']]):
            if not (type(b) == str):
                print(b)
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
        # select first token of sequence
        sentence_features = token_embeddings[:, 0, :]
        # project
        sentence_features = self.text_projection(sentence_features)
        # normalize
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)

        return sentence_features

    def training_step(self, batch, batch_idx):

        self.lr_scheduler_step(batch_idx)

        audio_embeddings, text_embeddings = self.forward(batch) # batch 1: sound IDs ['204046', '266329']; ['Paper_Parchment_Rustling.wav', 'metalTunnel.wav']

        if self.training_mode == 'bi-encoder':
            # 原有的双编码器对比损失逻辑
            C = torch.matmul(audio_embeddings, text_embeddings.T) / torch.abs(self.tau)
            C_audio = torch.log_softmax(C, dim=0)
            C_text = torch.log_softmax(C, dim=1)
            paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
            I = torch.tensor(paths[None, :] == paths[:, None]).to(self.device)
            loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())
        else:
            # Transformer 模式：引入负采样
            batch_size = audio_embeddings.size(0)
            
            # 1. 正样本得分
            pos_scores = self.forward_fusion(audio_embeddings, text_embeddings)
            
            # 2. 负样本采样：将文本循环移动一位，构造不匹配对
            text_embeds_neg = torch.roll(text_embeddings, shifts=1, dims=0)
            neg_scores = self.forward_fusion(audio_embeddings, text_embeds_neg)
            
            # 3. 组合得分与标签
            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            
            # 4. 使用二分类交叉熵损失 (BCEWithLogitsLoss 不需要对 score 预先做 sigmoid)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(all_scores, all_labels)

                


        # # compute pairwise similarities
        # C = torch.matmul(audio_embeddings, text_embeddings.T)

        # # scale cosine similarities with temperature < 1
        # # (otherwise $-1 <= C_{ij} <= 1$)
        # C = C / torch.abs(self.tau)

        # compute P(a|t) and P(t|a)
        # C_audio = torch.log_softmax(C, dim=0)
        # C_text = torch.log_softmax(C, dim=1)

        # # prediction target
        # paths = np.array([hash(batch['dataset'][i] + batch['subset'][i] + p) for i, p in enumerate(batch['fname'])])
        # I = torch.tensor(paths[None, :] == paths[:, None])

        # loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())

        self.log("train/loss", loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        # self.log('train/tau', torch.abs(self.tau), sync_dist=True)

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

        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        # audios in clotho can have five captions
        # this snippet discards every occurrence of a duplicate audio
        #
        target = [] # prediction targets for later
        select = [] # indices of the first occurrence for later
        first_occurrence = {} # temporary cache to keep track of first occurrences
        for i, p in enumerate(paths): # iterate over all paths
            index = first_occurrence.get(p)
            if index is None:  # First time seeing this path
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i) # these audios will be selected
            target.append(index) # all paths need a target - choose the correct one
        paths = paths[select]

        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]# only select unique audios
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        # concatenate global ranking
        C = torch.matmul(text_embeddings, audio_embeddings.T)

        # get top 10
        top_ten = C.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        # recall metrics
        r_1 = (top_ten[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5 = (top_ten[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10 = (top_ten == target[:, None]).sum(axis=1).mean()

        # mAP@10
        AP = 1 / ((top_ten == target[:, None]).argmax(axis=1) + 1)
        AP[~(top_ten == target[:, None]).any(axis=1)] = 0
        mAP = AP.mean()

        # log retrieval performance
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

            # index of query in C
            matched_files["query_index"] = matched_files["query"].transform(lambda x: captions.tolist().index(x))

            # new ground truth
            matched_files["new_audio_indices"] = matched_files["audio_filenames"].transform(lambda x: [paths.tolist().index(y) for y in x])
            matched_files["TP_ranks"] = matched_files.apply(lambda row: get_ranks(C[row["query_index"]], row["new_audio_indices"]), axis=1)

            def average_precision_at_k(relevant_ranks, k=10):
                relevant_ranks = sorted(relevant_ranks)
                ap = 0.0
                for i, rank in enumerate(relevant_ranks, start=1):
                    if rank >= k:
                        break
                    ap += i / (rank + 1) # precision at threshold
                return ap / len(relevant_ranks)  # Normalize by total number of relevant items

            new_mAP = matched_files["TP_ranks"].apply(lambda ranks: average_precision_at_k(ranks, 10)).mean()
            self.log(f'{prefix}_multiple_positives/mAP@10', new_mAP)
        # empty cached batches from validation loop
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
            # Linear warmup
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)
