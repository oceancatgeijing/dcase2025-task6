import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")



import os
from typing import Union, List, Mapping

import sys
import types
import torchaudio
from dataclasses import dataclass
import torch
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


# ==========================================================
# 针对 RTX 5090 (Blackwell) 的硬核兼容补丁
# ==========================================================
# 1. 强制设置可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

if torch.cuda.is_available():
    # 强制修改算力识别函数，让 Lightning 认为这是 sm_90
    # 这是解决 "Device should be GPU, got cpu instead" 的核心
    torch.cuda.get_device_capability = lambda _: (9, 0)
    
    # 针对 5090 性能优化：开启 bf16 支持识别
    torch.cuda.is_bf16_supported = lambda: True
    
    # 打印确认
    print(f"✅ 已探测到 {torch.cuda.get_device_name(0)}，强制映射为 sm_90 兼容模式")
# ==========================================================

import wandb

import argparse
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything


from aac_datasets import Clotho, WavCaps, AudioCaps
from torch.utils.data import DataLoader
from d25_t6.datasets.download_datasets import download_clotho, download_audiocaps, download_wavcaps_mp3
from d25_t6.datasets.audio_loading import custom_loading
from d25_t6.datasets.utils import exclude_broken_files, exclude_forbidden_files, exclude_forbidden_and_long_files
from d25_t6.datasets.batch_collate import CustomCollate
from d25_t6.datasets.wavcaps_local import WavCapsLocalDataset  # 导入本地 WavCaps

from d25_t6.retrieval_module import AudioRetrievalModel



def train(
        model: AudioRetrievalModel,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
):
    """
    Trains the AudioRetrievalModel using provided datasets, logger, and configuration arguments.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The model to be trained.
        train_ds (torch.utils.data.Dataset): The training dataset.
        val_ds (torch.utils.data.Dataset): The validation dataset.
        logger (Union[None, WandbLogger]): The logger for tracking training metrics.
        args (dict): A dictionary of configuration arguments for training.

    Returns:
        d25_t6.retrieval_module.AudioRetrievalModel: The trained model.
    """
    # get a unique experiment name for name of checkpoint
    if wandb.run is not None:
        experiment_name = wandb.run.name or wandb.run.id  # Use name if available, else use ID
    else:
        experiment_name = "experiment_" + wandb.util.generate_id()  # Random unique ID fallback

    # create path for the model checkpoints
    checkpoint_dir = os.path.join(args["checkpoints_path"], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}",
        save_top_k=1,
        monitor="val/mAP@10",
        mode="max",
        save_last=True
    )

    # 自动检测当前环境最强的加速器
    if torch.cuda.is_available():
        accelerator = "gpu"  # Windows/Linux (NVIDIA)
    elif torch.backends.mps.is_available():
        accelerator = "mps"  # macOS (Apple Silicon)
    else:
        accelerator = "cpu"  # 无 GPU 环境

    # trainer
    trainer = pl.Trainer(
        # accelerator=accelerator,
        accelerator="cuda",
        # devices=args['devices'],
        # devices=args['devices'] if args['devices'] != 'auto' else 1,
        devices=[0],
        # strategy="single_device",
        logger=logger if wandb.run else None,
        callbacks=[checkpoint_callback],
        max_epochs=args['max_epochs'],
        precision="32",
        gradient_clip_val=1.0,
        # precision="bf16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_ds, batch_size=args['batch_size'], num_workers=args['n_workers'], shuffle=True, drop_last=True,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        val_dataloaders=DataLoader(
            val_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        ckpt_path=args['resume_ckpt_path'] # should be none unless training is resumed
    )

    return model

def test(
        model: AudioRetrievalModel,
        test_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.
        test_ds (torch.utils.data.Dataset): The test dataset.
        logger (Union[None, WandbLogger]): The logger for tracking test metrics.
        args (dict): A dictionary of configuration arguments for testing.

    Returns:
        dict: The result of the model evaluation on the test dataset.
    """
    # trainer = pl.Trainer(
    #     devices=args['devices'],
    #     logger=logger if wandb.run else None,
    #     callbacks=None,
    #     max_epochs=args['max_epochs'],
    #     precision="16-mixed",
    #     num_sanity_val_steps=0,
    #     fast_dev_run=False
    # )
    trainer = pl.Trainer(
        # accelerator=accelerator,
        accelerator="cuda",
        # devices=args['devices'],
        # devices=args['devices'] if args['devices'] != 'auto' else 1,
        devices=[0],
        # strategy="single_device",
        logger=logger if wandb.run else None,
        # callbacks=[checkpoint_callback],
        max_epochs=args['max_epochs'],
        # precision="32",
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### test on the eval set
    result = trainer.test(
        model,
        DataLoader(
            test_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        )
    )

    return result


def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='auto', help='Device selection (e.g., auto, cpu, cuda, etc.)')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help='Compile the model if GPU version >= 7.')
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction, help='Log metrics in wandb or not.')

    # Parameter initialization & resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=21208, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--rampdown_epochs', type=int, default=15, help='Number of ramp-down epochs')
    parser.add_argument('--max_lr', type=float, default=2e-5, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-6, help='Minimum learning rate')
    parser.add_argument('--initial_tau', type=float, default=0.05, help='Initial tau value')
    parser.add_argument('--tau_trainable', default=False, action=argparse.BooleanOptionalAction, help='Temperature parameter is trainable or not.')

    # PaSST parameters
    parser.add_argument('--s_patchout_t', type=int, default=15, help='Temporal patchout size')
    parser.add_argument('--s_patchout_f', type=int, default=2, help='Frequency patchout size')

    # RoBERTa parameters
    parser.add_argument('--roberta_base', default=False, action=argparse.BooleanOptionalAction,  help='Use Roberta base or large.')

    # use additional data sets...
    parser.add_argument('--wavcaps', default=False, action=argparse.BooleanOptionalAction, help='Include WavCaps in the training or not.')
    parser.add_argument('--audiocaps', default=False, action=argparse.BooleanOptionalAction, help='Include AudioCaps in the training or not.')
    parser.add_argument('--ablate_clean_setup', default=True, action=argparse.BooleanOptionalAction, help='Include ClothoV2.1 eval, test in the training or not.')

    # 本地 WavCaps 参数
    parser.add_argument('--wavcaps_local', default=False, action=argparse.BooleanOptionalAction, 
                        help='使用本地 WavCaps CSV 标注（不通过 aac_datasets 下载）')
    parser.add_argument('--wavcaps_csv', type=str, 
                        default='/root/autodl-tmp/dcase2025/data/WavCaps_mp3/json_files/wavcaps_qwen3_omni_local_label.csv',
                        help='本地 WavCaps 标注 CSV 文件路径')
    parser.add_argument('--wavcaps_audio_root', type=str,
                        default='/root/autodl-tmp/dcase2025/data/WavCaps_mp3/Audio',
                        help='本地 WavCaps 音频文件根目录')

    # Paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset; dataset will be downloaded into this folder.')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints to.')

    # run training / test
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help='Run training or not.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction, help='Run testing or not.')

    parser.add_argument('--training_mode', type=str, default='bi-encoder', 
                        choices=['bi-encoder', 'transformer'],
                        help='选择训练方式: bi-encoder (双编码器) 或 transformer (融合编码器)')
    parser.add_argument('--temporal_aware', default=False, action=argparse.BooleanOptionalAction, 
                    help='启用时序感知模式：保留音频分段信息而非平均 (默认: False)')
    
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    """
    Entry point for training and testing the model.
    - Downloads datasets if necessary.
    - Initializes logging and model.
    - Runs training and/or testing based on arguments.
    """
    
    # 必须先解析参数，后续才能使用 args
    args = get_args()

    # 适配win和mac
    import platform

    if platform.system() == "Windows":
        # Windows 处理大量 workers 效率较低且易内存溢出，建议设置较小值
        default_workers = 4 
        # Windows 下 persistent_workers 必须配合 num_workers > 0 使用
        persistent_workers = args['n_workers'] > 0
    else:
        default_workers = 16
        persistent_workers = True

    os.makedirs(args["data_path"], exist_ok=True)
    # download data sets; will be ignored if exists
    # ClothoV2.1
    download_clotho(args["data_path"])
    # AudioCAps
    if args['audiocaps']:
        download_audiocaps(args["data_path"])
    # WavCaps (通过 aac_datasets 下载的)
    if args['wavcaps']:
        download_wavcaps_mp3(args["data_path"])
        # download_wavcaps(args["data_path"], args["huggingface_cache_path"])

    # set a seed to make experiments reproducible
    if args['seed'] > 0:
        seed_everything(args['seed'], workers=True)
    else:
        print("Not seeding experiment.")

    # initialize wandb, i.e., the logging framework
    if args['logging']:
        wandb.init(project="d25_t6")
        logger = WandbLogger()
    else:
        logger = None

    # initialize the model
    if args['load_ckpt_path']:
        model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'])
    else:
        model = AudioRetrievalModel(**args)

    # train
    if args['train']:
        # get training ad validation data sets; add the resampling transformation
        train_ds = custom_loading(Clotho(subset="dev", root=args["data_path"], flat_captions=True))

        if args['audiocaps']:
            ac = custom_loading(
                AudioCaps(subset="train", root=args["data_path"], download=True, download_audio=False, audio_format='mp3')
            )
            train_ds = torch.utils.data.ConcatDataset([train_ds, ac])

        # 标准 WavCaps（通过 aac_datasets）
        if args['wavcaps']:
            # load the subsets
            wc_f = exclude_forbidden_files(custom_loading(WavCaps(subset="freesound", root=args["data_path"])))
            wc_b = custom_loading(WavCaps(subset="bbc", root=args["data_path"]))
            wc_s = custom_loading(WavCaps(subset="soundbible", root=args["data_path"]))
            wc_a = exclude_broken_files(custom_loading(WavCaps(subset="audioset_no_audiocaps" if not args["ablate_clean_setup"] else "audioset", root=args["data_path"])))
            train_ds = torch.utils.data.ConcatDataset([train_ds, wc_f, wc_b, wc_s, wc_a])
        
        # 本地 WavCaps（通过 CSV 加载）
        if args['wavcaps_local']:
            print(f"Loading local WavCaps from {args['wavcaps_csv']}")
            wavcaps_local_ds = WavCapsLocalDataset(
                csv_path=args['wavcaps_csv'],
                audio_root=args['wavcaps_audio_root'],
                target_sample_rate=32000,
                max_length=30,
                flat_captions=True,  # 展开为独立样本，增加多样性
                sample_ratio=0.1
            )
            wavcaps_local_ds = custom_loading(wavcaps_local_ds)
            train_ds = torch.utils.data.ConcatDataset([train_ds, wavcaps_local_ds])
            print(f"Added {len(wavcaps_local_ds)} local WavCaps samples")

        val_ds = custom_loading(Clotho(subset="val", root=args["data_path"], flat_captions=True))

        model = train(model, train_ds, val_ds, logger, args)

    # test
    if args['test']:
        test_ds = custom_loading(Clotho(subset="eval", root=args["data_path"], flat_captions=True))

        results = test(model, test_ds, logger, args)
        print(results)