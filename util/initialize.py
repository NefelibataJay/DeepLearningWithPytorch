import torch
import numpy as np
from util.tokenize import REGISTER_TOKENIZER
from datasets import REGISTER_DATASET
import random


def init_config(config):
    init_seed(config.seed)
    tokenizer = init_tokenizer(config)


def init_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_tokenizer(config):
    return REGISTER_TOKENIZER[config.tokenizer_name](config)


def init_dataloader(config, tokenizer, stage='train'):
    datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage=stage)


def init_model(config):
    pass


def init_optimizer(optimizer_name):
    pass


def init_scheduler(scheduler_name):
    pass


def init_loss(loss_name):
    pass


def init_metric(metric_name):
    pass
