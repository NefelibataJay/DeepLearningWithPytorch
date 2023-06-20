import argparse
import os

import torch
from omegaconf import OmegaConf

from tool.trainer.conformerctctrainer import ConformerCTCTrainer
from util.initialize import init_config


def main():
    args = get_args()
    config = get_config(args.config_path)

    config.dataset.dataset_path = args.dataset_path
    config.dataset.manifest_path = args.manifest_path
    config.save_path = args.save_path

    if args.stage == "train":
        model, tokenizer, optimizer, scheduler, criterion, metric, train_dataloader, valid_dataloader, = init_config(
            config, stage="train")
        if args.checkpoint_path is not None:
            model.load_state_dict(torch.load(args.checkpoint_path))

        # train model
        trainer = ConformerCTCTrainer(config, tokenizer, model, optimizer, scheduler, criterion, metric, args.device)
        trainer.train(train_dataloader, valid_dataloader)

        # final save model
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        torch.save(model.state_dict(), os.path.join(config.save_path, f"{config.model_name}_final.pt"))
    else:
        # TODO test model
        pass


def get_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="conf/config.yaml", help="config path")
    parser.add_argument("--dataset_path", default=" ", help="dataset path")
    parser.add_argument("--manifest_path", default="../manifests/aishell_chars", help="manifest path")
    parser.add_argument("--checkpoint_path", default=None, help="checkpoint path")
    parser.add_argument("--save_path", default=None, help="save path")
    parser.add_argument("--stage", default="train", help="stage")
    parser.add_argument("--device", default="cuda", help="device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
