import torch
import numpy as np
import random

from tool.search import REGISTER_SEARCH
from tool.tokenize import REGISTER_TOKENIZER
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
    metric = init_metric(config)

    if stage == 'train':
        train_dataloader, valid_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, optimizer, scheduler, metric, train_dataloader, valid_dataloader,
    else:
        test_dataloader = init_dataloader(config, tokenizer, stage=stage)
        return model, tokenizer, optimizer, scheduler, metric, test_dataloader


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
        train_dataloader = DataLoader(train_datasets, shuffle=True, collate_fn=_collate_fn, **config.dataloader)
        valid_dataloader = DataLoader(valid_datasets, shuffle=True, collate_fn=_collate_fn, **config.dataloader)
        return train_dataloader, valid_dataloader
    else:
        test_datasets = REGISTER_DATASET[config.dataset_name](config, tokenizer, stage='test')
        test_dataloader = DataLoader(test_datasets, batch_size=5, shuffle=False, collate_fn=_collate_fn)
        return test_dataloader


def init_model(config):
    model = REGISTER_MODEL[config.model_name](config)
    # TODO add weight init here if you want
    # TODO load pretrained model, load weight here
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p)
    print(model)
    return model


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
    metric = REGISTERED_METRICS[config.metric_name](**config.metric)
    return metric


def init_search(config):
    search = REGISTER_SEARCH[config.search_name](**config.search)
    return search

def _collate_fn(batch):
    inputs = [i[0] for i in batch]

    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = [torch.IntTensor(i[2]) for i in batch]
    target_lengths = torch.IntTensor([i[3] for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths

