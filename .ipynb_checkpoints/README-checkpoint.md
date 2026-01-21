# DCASE2025 - Task 6 - Baseline System

**Task Organizers**: 
- *Huang Xie* (Tampere University)
- *Tuomas Virtanen* (Tampere University)
- *Benno Weck* (Universitat Pompeu Fabra)
- *Paul Primus* (Johannes Kepler University Linz)

Repository Contact: paul.primus@jku.at


## Language-Based Audio Retrieval

Language-Based Audio Retrieval focuses on the development of audio retrieval systems that can find audio recordings based on textual queries. 

While similar to previous editions, this year's evaluation setup introduces the possibility of multiple matching audio candidates for a single query.

To support this, we provide additional correspondence annotations for audio-query pairs in the evaluation sets, enabling a more nuanced assessment of the retrieval systems' performance during development and final ranking.

**The additional annotations for the evaluation are now available, and the results are reported in the results table below.**

## Baseline System

This repository contains the code for the baseline system of the DCASE 2025 Challenge Task 6.

* The training loop is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). 
* Logging is implemented using [Weights and Biases](https://wandb.ai/site). 
* The architecture of the baseline system is based on the top-ranked system of [Task 8 in the DCASE 2024 challenge](https://dcase.community/challenge2024/task-language-based-audio-retrieval-results).
* It uses the [Patch out Fast Spectrogram Transformer](https://arxiv.org/abs/2110.05069) (PaSST) and [RoBERTa](https://arxiv.org/abs/1907.11692)-large to encode audio recordings and text-queries.
* Datasets are loaded via [aac-datasets](https://github.com/Labbeti/aac-datasets).

## Getting Started

Prerequisites
- linux (tested on Ubuntu 24.04)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install), e.g., [Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

1. Clone this repository.

```
git clone https://github.com/CPJKU/dcase2025_task6_baseline.git
```

2. Create and activate a conda environment with Python 3.11:

```
conda create -n d25_t6 python=3.11
conda activate d25_t6
```

3. Install 7z

```
# (on linux)
sudo apt install p7zip-full
# (on linux)
conda install -c conda-forge p7zip
# (on windows)
conda install -c conda-forge 7zip
```


4. Install a [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

```
# for cuda >= 12.1 (check with nvidia-smi)
pip3 install torch torchvision torchaudio
# for cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# for otther versions see: https://pytorch.org/get-started/locally/
```

5. Install other dependencies:
```
pip3 install -r requirements.txt
```

6. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.

## Run Experiments


The default project structure is:
```
d25_t6/                               # Baseline implementation
checkpoints/                          # Model checkpoints (change with --checkpoints_path)
data/                                 # Dataset storage (change with --data_path)
│
├── CLOTHO_v2.1/        (20.5  GB)
├── AUDIOCAPS/          (2.8   GB)
└── WavCaps/            (156.9 GB)
resources/                            # Miscellaneous reference files
│
└── dcase2025_task6_excl...           # List of sounds excluded from training
└── example_predictions.csv           # Example predition file (for the submission package)
scripts/                              # Utility scripts
│
└── convert_flac_to_mp3.py            # Converts WavCaps to mp3 to save memory (only required if you download WavCaps from HuggingFace)
README.md                             # Project overview and setup instructions
```

Run `python -m d25_t6.train --help` to see all command line options.

The training procedure can be started on a by running the following command:

**NVIDA A40** 
```
python -m d25_t6.train --compile --data_path=data --seed=13
```

**NVIDIA 2080Ti**
```
python -m d25_t6.train --data_path=data --batch_size=4 --batch_size_eval=4 --no-compile --max_lr=5e-6 --seed=13
```

Running the training script automatically downloads the ClothoV2.1 dataset into the folder specified in `--data_path`

To include AudioCaps in the training use (data will be downloaded automatically):
```
python -m d25_t6.train --audiocaps --data_path=data --seed=492412 --compile --no-tau_trainable
```

To include AudioCaps and WavCaps in training use (data will be downloaded automatically):
```
python -m d25_t6.train --audiocaps --wavcaps --data_path=data --seed=967251 --compile --no-tau_trainable
```

Use options `--no-train --no-test` to just download the data sets.


## Baseline Results

The primary evaluation metric of previous editions of this task was the **mean average precision at 10**. 
This year, the focus is on mAP@16 on the Clotho dataset with additional annotations.

For exact details on how the submissions will be ranked consult the [official task description](https://dcase.community/challenge2025/). 

The table below also provides the recall at 1, 5 and 10.

|                   **trained on** | **mAP@16** (new annotations) | **mAP@10** | **R@1** | **R@5** | **R@10** | trained on GPU     | avg. runtime |
|---------------------------------:|:-----------------------------|:-----------|:--------|:--------|:---------|:-------------------|:-------------|
|                     **ClothoV2** | 32.82                        | 27.83      | 16.95   | 42.46   | 55.80    | NVIDIA A40         | 2h 16m       |
|                     **ClothoV2** | 34.59                        | 28.76      | 16.05      | 42.73   | 58.04    | NVIDIA RTX 2080 Ti | 8h 53m       |
|            **Clotho, AudioCaps** | 35.21                        | 30.97      | 19.56      | 46.45   | 59.48    | NVIDIA A40         | 7h 16m       |
| **Clotho, AudioCaps, WavCaps**   | 40.59                        | 35.23      | 23.29      | 52.17   | 64.78    | NVIDIA A40         | 34h 44m      |

A checkpoint of the model trained on Clotho, AudioCaps, WavCaps is available [here](https://cloud.cp.jku.at/index.php/s/6ZTQ3mcwk9AAS4i).

### Create Predictions

To create predictions for a specific checkpoint run:
```
python -m d25_t6.predict \
--load_ckpt_path=PATH_OF_CHECKPOINT_FILE.ckpt \
--retrieval_audio_path=PATH_OF_FOLDER_CONTAINING_AUDIOS \
--retrieval_captions=PATH_OF_CSV_LISTING_QUERIS.csv \
--predictions_path=PATH_TO_WHERE_PREDICTIONS_WILL_BE_STORED
```
# Citation
If you use this repository, cite our related paper:
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

# References
If you use this repository, please cite the following related work that it builds on:
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

# License & Citation

Only academic uses are allowed for WavCaps and AudioCaps dataset. By downloading audio clips through the links provided, you agree that you will use the audios for research purposes only. For credits for audio clips from FreeSound, please refer to its own page.

For detailed license information, please refer to: 
- [Clotho](https://zenodo.org/records/4783391)
- [AudioCaps](https://audiocaps.github.io/)
- [FreeSound](https://freesound.org/help/faq/#licenses)
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk/licensing)
- [SoundBible](https://soundbible.com/about.php)
