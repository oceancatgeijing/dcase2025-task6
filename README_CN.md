# DCASE2025 - Task 6 - 基线系统

**任务组织者**：
- *Huang Xie* (坦佩雷大学)
- *Tuomas Virtanen* (坦佩雷大学)
- *Benno Weck* (庞培法布拉大学)
- *Paul Primus* (林茨约翰内斯开普勒大学)

仓库联系：paul.primus@jku.at

## 基于语言的音频检索

基于语言的音频检索专注于开发能够根据文本查询找到音频录音的音频检索系统。

虽然与之前的版本类似，但今年的评估设置引入了单个查询可能匹配多个音频候选的可能性。

为了支持这一点，我们在评估集中为音频-查询对提供了额外的对应关系注释，使得在开发和最终排名期间能够更细致地评估检索系统的性能。

**评估的额外注释现已可用，结果在下面的结果表中报告。**

## 基线系统

本仓库包含 DCASE 2025 挑战赛 Task 6 的基线系统代码。

* 训练循环使用 [PyTorch](https://pytorch.org/) 和 [PyTorch Lightning](https://lightning.ai/) 实现。
* 日志记录使用 [Weights and Biases](https://wandb.ai/site) 实现。
* 基线系统的架构基于 [DCASE 2024 挑战赛 Task 8](https://dcase.community/challenge2024/task-language-based-audio-retrieval-results) 的顶级系统。
* 它使用 [Patch out Fast Spectrogram Transformer](https://arxiv.org/abs/2110.05069) (PaSST) 和 [RoBERTa](https://arxiv.org/abs/1907.11692)-large 来编码音频录音和文本查询。
* 数据集通过 [aac-datasets](https://github.com/Labbeti/aac-datasets) 加载。

## 快速开始

### 前置要求
- linux (已在 Ubuntu 24.04 上测试)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install)，例如 [Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

1. 克隆本仓库。

```
git clone https://github.com/CPJKU/dcase2025_task6_baseline.git
```

2. 创建并激活一个 Python 3.11 的 conda 环境：

```
conda create -n d25_t6 python=3.11
conda activate d25_t6
```

3. 安装 7z

```
# (在 linux 上)
sudo apt install p7zip-full
# (在 linux 上)
conda install -c conda-forge p7zip
# (在 windows 上)
conda install -c conda-forge 7zip
```

4. 安装适合您系统的 [PyTorch](https://pytorch.org/get-started/previous-versions/) 版本。例如：

```
# 适用于 cuda >= 12.1 (使用 nvidia-smi 检查)
pip3 install torch torchvision torchaudio
# 适用于 cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 其他版本请参见：https://pytorch.org/get-started/locally/
```

5. 安装其他依赖：
```
pip3 install -r requirements.txt
```

6. 如果您之前没有使用过 [Weights and Biases](https://wandb.ai/site) 进行日志记录，您可以创建一个免费账户。在您的机器上运行 ```wandb login```，并从[此](https://wandb.ai/authorize)链接复制您的 API 密钥到命令行。

## 运行实验

默认的项目结构是：
```
d25_t6/                               # 基线实现
checkpoints/                          # 模型检查点 (使用 --checkpoints_path 更改)
data/                                 # 数据集存储 (使用 --data_path 更改)
│
├── CLOTHO_v2.1/        (20.5  GB)
├── AUDIOCAPS/          (2.8   GB)
└── WavCaps/            (156.9 GB)
resources/                            # 其他参考文件
│
└── dcase2025_task6_excl...           # 训练中排除的声音列表
└── example_predictions.csv           # 示例预测文件 (用于提交包)
scripts/                              # 工具脚本
│
└── convert_flac_to_mp3.py            # 将 WavCaps 转换为 mp3 以节省内存 (仅当您从 HuggingFace 下载 WavCaps 时需要)
README.md                             # 项目概述和设置说明
```

运行 `python -m d25_t6.train --help` 查看所有命令行选项。

可以通过运行以下命令启动训练过程：

**NVIDA A40**
```
python -m d25_t6.train --compile --data_path=data --seed=13
```

**NVIDIA 2080Ti**
```
python -m d25_t6.train --data_path=data --batch_size=4 --batch_size_eval=4 --no-compile --max_lr=5e-6 --seed=13
```

运行训练脚本会自动将 ClothoV2.1 数据集下载到 `--data_path` 指定的文件夹中

要在训练中包含 AudioCaps，请使用（数据将自动下载）：
```
python -m d25_t6.train --audiocaps --data_path=data --seed=492412 --compile --no-tau_trainable
```

要在训练中包含 AudioCaps 和 WavCaps，请使用（数据将自动下载）：
```
python -m d25_t6.train --audiocaps --wavcaps --data_path=data --seed=967251 --compile --no-tau_trainable
```

使用选项 `--no-train --no-test` 仅下载数据集。

## 基线结果

本任务之前版本的主要评估指标是 **平均精度均值 @10**。
今年，重点是使用额外注释在 Clotho 数据集上的 mAP@16。

有关提交如何排名的确切详细信息，请查阅[官方任务描述](https://dcase.community/challenge2025/)。

下表还提供了召回率 @1、@5 和 @10。

| **训练数据集**                 | **mAP@16** (新注释) | **mAP@10** | **R@1** | **R@5** | **R@10** | 训练 GPU           | 平均运行时间 |
| ------------------------------ | ------------------- | ---------- | ------- | ------- | -------- | ------------------ | ------------ |
| **ClothoV2**                   | 32.82               | 27.83      | 16.95   | 42.46   | 55.80    | NVIDIA A40         | 2h 16m       |
| **ClothoV2**                   | 34.59               | 28.76      | 16.05   | 42.73   | 58.04    | NVIDIA RTX 2080 Ti | 8h 53m       |
| **Clotho, AudioCaps**          | 35.21               | 30.97      | 19.56   | 46.45   | 59.48    | NVIDIA A40         | 7h 16m       |
| **Clotho, AudioCaps, WavCaps** | 40.59               | 35.23      | 23.29   | 52.17   | 64.78    | NVIDIA A40         | 34h 44m      |

在 Clotho、AudioCaps、WavCaps 上训练的模型检查点可在[此处](https://cloud.cp.jku.at/index.php/s/6ZTQ3mcwk9AAS4i)获取。

### 创建预测

要为特定检查点创建预测，请运行：
```
python -m d25_t6.predict \
--load_ckpt_path=检查点文件路径.ckpt \
--retrieval_audio_path=包含音频的文件夹路径 \
--retrieval_captions=列出查询的CSV文件路径.csv \
--predictions_path=预测结果存储路径
```

# 引用

如果您使用本仓库，请引用我们的相关论文：
```
@inproceedings{Primus2024,
    author = "Primus, Paul and Schmid, Florian and Widmer, Gerhard",
    title = "Estimated Audio–Caption Correspondences Improve Language-Based Audio Retrieval",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    address = "Tokyo, Japan",
    month = "October",
    year = "2024",
    pages = "121--125",
    abstract = "Dual-encoder-based audio retrieval systems are commonly optimized with contrastive learning on a set of matching and mismatching audio–caption pairs. This leads to a shared embedding space in which corresponding items from the two modalities end up close together. Since audio–caption datasets typically only contain matching pairs of recordings and descriptions, it has become common practice to create mismatching pairs by pairing the audio with a caption randomly drawn from the dataset. This is not ideal because the randomly sampled caption could, just by chance, partly or entirely describe the audio recording. However, correspondence information for all possible pairs is costly to annotate and thus typically unavailable; we, therefore, suggest substituting it with estimated correspondences. To this end, we propose a two-staged training procedure in which multiple retrieval models are first trained as usual, i.e., without estimated correspondences. In the second stage, the audio–caption correspondences predicted by these models then serve as prediction targets. We evaluate our method on the ClothoV2 and the AudioCaps benchmark and show that it improves retrieval performance, even in a restricting self-distillation setting where a single model generates and then learns from the estimated correspondences. We further show that our method outperforms the current state of the art by 1.6 pp. mAP@10 on the ClothoV2 benchmark."
}
```

# 参考文献

如果您使用本仓库，请引用以下相关研究工作：
```
@inproceedings{PaSST,
  author       = {Khaled Koutini and
                  Jan Schl{\"{u}}ter and
                  Hamid Eghbal{-}zadeh and
                  Gerhard Widmer},
  title        = {Efficient Training of Audio Transformers with Patchout},
  booktitle    = {Interspeech 2022, 23rd Annual Conference of the International Speech
                  Communication Association, Incheon, Korea, 18-22 September 2022},
  pages        = {2753--2757},
  publisher    = {{ISCA}},
  year         = {2022},
  url          = {https://doi.org/10.21437/Interspeech.2022-227},
  doi          = {10.21437/Interspeech.2022-227},
}
```
```
@inproceedings{Clotho,
  author       = {Konstantinos Drossos and
                  Samuel Lipping and
                  Tuomas Virtanen},
  title        = {Clotho: an Audio Captioning Dataset},
  booktitle    = {2020 {IEEE} International Conference on Acoustics, Speech and Signal
                  Processing, {ICASSP} 2020, Barcelona, Spain, May 4-8, 2020},
  pages        = {736--740},
  publisher    = {{IEEE}},
  year         = {2020},
  url          = {https://doi.org/10.1109/ICASSP40776.2020.9052990},
  doi          = {10.1109/ICASSP40776.2020.9052990},
 }
```
```
@inproceedings{AudioCaps,
  author       = {Chris Dongjoo Kim and
                  Byeongchang Kim and
                  Hyunmin Lee and
                  Gunhee Kim},
  editor       = {Jill Burstein and
                  Christy Doran and
                  Thamar Solorio},
  title        = {AudioCaps: Generating Captions for Audios in The Wild},
  booktitle    = {Proceedings of the 2019 Conference of the North American Chapter of
                  the Association for Computational Linguistics: Human Language Technologies,
                  {NAACL-HLT} 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long
                  and Short Papers)},
  pages        = {119--132},
  publisher    = {Association for Computational Linguistics},
  year         = {2019},
  url          = {https://doi.org/10.18653/v1/n19-1011},
  doi          = {10.18653/V1/N19-1011}
}
```
```
@article{DBLP:journals/corr/abs-1907-11692,
  author       = {Yinhan Liu and
                  Myle Ott and
                  Naman Goyal and
                  Jingfei Du and
                  Mandar Joshi and
                  Danqi Chen and
                  Omer Levy and
                  Mike Lewis and
                  Luke Zettlemoyer and
                  Veselin Stoyanov},
  title        = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal      = {CoRR},
  volume       = {abs/1907.11692},
  year         = {2019},
  url          = {http://arxiv.org/abs/1907.11692},
  eprinttype    = {arXiv},
  eprint       = {1907.11692},
}
```

```
@article{WavCaps,
  author       = {Xinhao Mei and
                  Chutong Meng and
                  Haohe Liu and
                  Qiuqiang Kong and
                  Tom Ko and
                  Chengqi Zhao and
                  Mark D. Plumbley and
                  Yuexian Zou and
                  Wenwu Wang},
  title        = {WavCaps: {A} ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset
                  for Audio-Language Multimodal Research},
  journal      = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  volume       = {32},
  pages        = {3339--3354},
  year         = {2024},
  url          = {https://doi.org/10.1109/TASLP.2024.3419446},
  doi          = {10.1109/TASLP.2024.3419446},
 }
```

# 许可与引用

WavCaps 和 AudioCaps 数据集仅允许学术用途。通过提供的链接下载音频片段，即表示您同意仅将音频用于研究目的。有关来自 FreeSound 的音频片段的信用，请参阅其自己的页面。

有关详细的许可信息，请参阅：
- [Clotho](https://zenodo.org/records/4783391)
- [AudioCaps](https://audiocaps.github.io/)
- [FreeSound](https://freesound.org/help/faq/#licenses)
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/licensing)
- [SoundBible](https://soundbible.com/about.php)

