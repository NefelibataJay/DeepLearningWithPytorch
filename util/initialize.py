import torch
import numpy as np
import random

from tool.search import REGISTER_SEARCH
from tool.tokenize import REGISTER_TOKENIZER
from datasets import REGISTER_DATASET
from tool.optimizer import REGISTER_OPTIMIZER, REGISTER_SCHEDULER
from models import REGISTER_MODEL
from torch.utils.data import DataLoaders
from tool.metrics import REGISTERED_METRICS
from util.collator import *


def init_config(config, stage='train', init_params=None):
    init_seed(config.seed)
    tokenizer = init_tokenizer(config)
    num_classes = len(tokenizer)
    config.model.num_classes = num_classes
    model = init_model(config, init_params)

    if stage == 'train':
        optimizer = init_optimizer(model, config)
        scheduler = init_scheduler(optimizer, config)
        train_dataloader, valid_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, optimizer, scheduler, train_dataloader, valid_dataloader,
    else:
        test_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, test_dataloader


def init_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def init_tokenizer(config):
    return REGISTER_TOKENIZER[config.tokenizer_name](config)


def init_dataloader(config, tokenizer, stage='train'):
    if stage == 'train':
        train_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='train')
        valid_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='valid')
        train_dataloader = DataLoader(train_datasets, shuffle=True, collate_fn=asr_collate_fn, **config.dataloader)
        valid_dataloader = DataLoader(valid_datasets, shuffle=False, collate_fn=asr_collate_fn, **config.dataloader)
        return train_dataloader, valid_dataloader
    else:
        test_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='test')
        test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, collate_fn=asr_collate_fn)
        test_dataloader.spec_aug = None
        return test_dataloader


def init_model(config, init_params):
    model = REGISTER_MODEL[config.model_name](config)
    # TODO add weight init here if you want
    # TODO load pretrained model, load weight here
    if init_params is not None:
        init_param(model, init_params)
    print(model)
    return model


def init_param(model, init_params):
    for p in model.parameters():
        pass


def init_optimizer(model, config):
    optimizer = REGISTER_OPTIMIZER[config.optimizer_name](model.parameters(), **config.optimizer)
    return optimizer


def init_scheduler(optimizer, config):
    if config.scheduler_name == 'gradual_warmup_lr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_scheduler.T_max)
        lr_scheduler = REGISTER_SCHEDULER[config.scheduler_name](optimizer,
                                                                 multiplier=config.lr_scheduler.multiplier,
                                                                 total_epoch=config.lr_scheduler.warmup_epochs,
                                                                 after_scheduler=scheduler)
    else:
        lr_scheduler = REGISTER_SCHEDULER[config.scheduler_name](optimizer, **config.lr_scheduler)
    return lr_scheduler


def init_metric(config):
    metric = REGISTERED_METRICS[config.metric_name]()
    return metric


def init_search(config):
    search = REGISTER_SEARCH[config.search_name](**config.search)
    return search
