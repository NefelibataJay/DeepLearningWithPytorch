import torch
import numpy as np
import random
from util.tokenize import REGISTER_TOKENIZER
from datasets import REGISTER_DATASET
from tool.optimizer import REGISTER_OPTIMIZER, REGISTER_SCHEDULER
from tool.loss import REGISTER_LOSS
from models import REGISTER_MODEL
from torch.utils.data import DataLoader
from tool.metrics import REGISTERED_METRICS


def init_config(config, stage='train'):
    init_seed(config.seed)
    tokenizer = init_tokenizer(config)

    num_classes = len(tokenizer)
    config.model.num_classes = num_classes

    model = init_model(config)
    optimizer = init_optimizer(model, config)
    scheduler = init_scheduler(optimizer, config)
    criterion = init_criterion(config)
    metric = init_metric(config)

    if stage == 'train':
        train_dataloader, valid_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, optimizer, scheduler, criterion, metric, train_dataloader, valid_dataloader,
    else:
        test_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, optimizer, scheduler, criterion, metric, test_dataloader


def init_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_tokenizer(config):
    return REGISTER_TOKENIZER[config.tokenizer_name](config)


def init_dataloader(config, tokenizer, stage='train'):
    if stage == 'train':
        train_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='train')
        valid_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='valid')
        train_dataloader = DataLoader(train_datasets, shuffle=True, **config.dataloader)
        valid_dataloader = DataLoader(valid_datasets, shuffle=True, **config.dataloader)
        return train_dataloader, valid_dataloader
    else:
        test_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='test')
        test_dataloader = DataLoader(test_datasets, batch_size=config.dataloader.batch_size, shuffle=False)
        return test_dataloader


def init_model(config):
    model = REGISTER_MODEL[config.model_name](config)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    print(model)
    return model


def init_optimizer(model, config):
    optimizer = REGISTER_OPTIMIZER[config.optimizer_name](model.parameters(), **config.optimizer)
    return optimizer


def init_scheduler(optimizer, config):
    lr_scheduler = REGISTER_SCHEDULER[config.scheduler_name](optimizer, **config.lr_scheduler)
    return lr_scheduler


def init_criterion(config):
    criterion = REGISTER_LOSS[config.loss_name](**config.loss)
    return criterion


def init_metric(config):
    metric = REGISTERED_METRICS[config.metric_name](**config.metric)
    return metric


def init_search():
    pass
