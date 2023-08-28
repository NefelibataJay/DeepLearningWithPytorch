#!/usr/bin/env python3
# encoding: utf-8
import math
import sys
import os

import argparse
import json
import codecs

import numpy as np
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader

torchaudio.set_audio_backend("soundfile")


class CollateFunc(object):
    def __init__(self, feat_dim, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = torchaudio.backend.soundfile_backend.info(wav_path).sample_rate
            resample_rate = sample_rate

            waveform, sample_rate = torchaudio.load(item[1])

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)

            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.feat_dim,
                              dither=0.0,
                              energy_floor=0.0,
                              sample_frequency=resample_rate)
            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)  # sum of square
            number += mat.shape[0]  # number of frames
        return number, mean_stat, var_stat


class AudioDataset(Dataset):
    def __init__(self, dataset_path, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                wav_path = line.strip().split("\t")[0]
                wav_path = os.path.join(dataset_path, wav_path)
                self.items.append(wav_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def compute_cmvn(configs):
    print("=============================================================")
    print("compute cmvn stats")

    feat_dim = configs.dataset.extractor_conf.num_mel_bins
    resample_rate = configs.dataset.sample_rate
    dataset_path = configs.dataset.dataset_path
    manifests_path = configs.dataset.manifests_path

    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioDataset(dataset_path, manifests_path)
    batch_size = 20
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             collate_fn=collate_func)
    with torch.no_grad():
        num_frames = 0
        means = torch.zeros(feat_dim)
        variance = torch.zeros(feat_dim)
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            means += mean_stat
            variance += var_stat
            num_frames += number

    for i in range(len(means)):
        means[i] /= num_frames
        variance[i] = variance[i] / num_frames - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / torch.sqrt(variance[i])

    print(f"means: {means} --- variance: {variance}")



if __name__ == "__main__":
    pass
