#!/usr/bin/env python3
# encoding: utf-8
import sys
import os

import argparse
import json
import codecs
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='train_config.yaml',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default="train/wav.scp", help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='train/global_cmvn',
                        help='global cmvn file')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf']['resample_rate']
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioDataset(args.in_scp)
    batch_size = 20
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        """
        all_number  所有语音帧数
        all_mean_stat  
        all_var_stat  
        """
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
