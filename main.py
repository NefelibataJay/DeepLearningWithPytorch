import argparse
import os
from omegaconf import OmegaConf

from util.initialize import init_config


def main():
    args = get_args()
    config = get_config(os.path.join(args.config_path, args.config_name + ".yaml"))

    model, = init_config(config)


def get_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="conf", help="config path")
    parser.add_argument("--config_name", default="config", help="config name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
