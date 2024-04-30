import os

import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from utils.config import Config
from model import Model
from dataset import KVPDataset

torch.set_float32_matmul_precision('medium')

def main():
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/kvp.yaml', help='path to train config file')
    parser.add_argument('--save_path', type=str, required=True, default=None, help='path where to save the model checkpoints')
    parser.add_argument('--train_folder', type=str, required=True, default=None, help='path to train data folder')
    parser.add_argument("--cache_dir", type=str, required=False, default='/tmp', help='path to transformer cache_dir')

    input_args = parser.parse_args()
    args = Config(input_args.config)
    args.save_path = input_args.save_path
    args.datasets_tess_common_gt_train_root = input_args.train_folder
    args.tokenizer_cache_dir = input_args.cache_dir
    args.llm_cache_dir = input_args.cache_dir
    args.trainer_default_root_dir = args.save_path

    print(vars(args))

    # create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # seed
    pl.seed_everything(args.seed)

    # model
    model = Model(args)

    # dataloaders
    train_dataset = KVPDataset(dataset_path=args.datasets_tess_common_gt_train_root, output_format=args.model_output_format)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_path, save_last=True, save_top_k=1)

    # trainer
    trainer = pl.Trainer(**args.trainer, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, ckpt_path='last')


if __name__ == '__main__':
    main()
