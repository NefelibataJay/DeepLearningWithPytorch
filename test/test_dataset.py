import torch
from omegaconf import OmegaConf
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from datasets.asr_dataset import ASRDataset
from tool.tokenize import ChineseCharTokenizer


def get_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


def _collate_fn(batch):
    inputs = [i[0] for i in batch]

    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = [torch.IntTensor(i[2]) for i in batch]
    target_lengths = torch.IntTensor([i[3] for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths


def test_dataset():
    config = get_config("../conf/config.yaml")
    tokenizer = ChineseCharTokenizer(config)
    train_dataset = ASRDataset(config, stage="train", tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                                                   collate_fn=_collate_fn)
    for batch in train_dataloader:
        speech_feature, input_lengths, transcript, target_lengths = batch
        print(speech_feature.shape)
        print(input_lengths)
        print(transcript)
        print(target_lengths)
        break
