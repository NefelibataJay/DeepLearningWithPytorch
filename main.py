import argparse
import torch
from omegaconf import OmegaConf
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from trainer.inferencer import SpeechToText
from trainer.trainer import Trainer
from util.initialize import init_config


def main():
    args = get_args()
    device = args.device
    config = get_config(args.config_path)

    config.dataset.dataset_path = args.dataset_path
    config.dataset.manifest_path = args.manifest_path
    config.save_path = args.save_path

    assert args.stage in ["train", "test"], "stage must be train or test"

    if args.stage == "train":
        model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader = init_config(
            config, stage="train")
        if args.checkpoint_path is not None:
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))

        # train model
        trainer = Trainer(config, tokenizer, model, optimizer, scheduler, device)
        trainer.train(train_dataloader, valid_dataloader)

        # final save model
        torch.save(model.state_dict(), os.path.join(config.save_path, "checkpoints", f"{config.model_name}_final.pt"))

    if args.stage == "test":
        assert args.checkpoint_path is not None, "checkpoint path must be not None"
        model, tokenizer, test_dataloader = init_config(config, stage="test")

        model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))
        model.eval()

        inference = SpeechToText(config, tokenizer, model, device)
        inference.recognition(test_dataloader)


def get_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="conf/config.yaml", help="config path")
    parser.add_argument("--dataset_path", default="./data_aishell", help="dataset path")
    parser.add_argument("--manifest_path", default="./manifests/aishell", help="manifest path")
    parser.add_argument("--checkpoint_path", default=None, help="checkpoint path")
    parser.add_argument("--save_path", default="./outputs", help="save path")
    parser.add_argument("--stage", default="train", help="stage")
    parser.add_argument("--device", default="cuda", help="device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
