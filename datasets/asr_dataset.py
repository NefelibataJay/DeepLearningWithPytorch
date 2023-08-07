import os

import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset

from tool.data_augmentations.speech_augment import SpecAugment
from tool.feature_extractor.speech_feature_extractor import AudioFeatureExtractor
from tool.tokenize.tokenizer import Tokenizer


class ASRDataset(Dataset):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer, stage):
        super(ASRDataset, self).__init__()
        self.tokenizer = tokenizer
        self.configs = configs.dataset
        self.sample_rate = self.configs.sample_rate
        self.manifest_path = self.configs.manifest_path
        self.dataset_path = self.configs.dataset_path

        self.feature_types = self.configs.feature_types
        self.extract_feature = AudioFeatureExtractor(feature_type=self.feature_types, sample_rate=self.sample_rate,
                                                     **self.configs.extractor_conf)

        assert stage in ["train", "valid", "test"]
        self.stage = stage
        self.audio_paths, self.transcripts = self._parse_manifest()

        if self.configs.speed_perturb:
            # 1.0:60%  0.9:20% 1.1:20%
            self.speed_perturb = torchaudio.transforms.SpeedPerturbation(self.sample_rate, [0.9, 1.0, 1.1, 1.0, 1.0])

        if self.configs.spec_aug:
            self.spec_aug = SpecAugment(**self.configs.spec_aug_conf)

    def __getitem__(self, idx):
        speech_feature = self._parse_audio(self.audio_paths[idx])  # [time, dim]
        transcript = self._parse_transcript(self.transcripts[idx])
        input_lengths = speech_feature.size(0)  # time
        target_lengths = len(transcript)

        return speech_feature, input_lengths, transcript, target_lengths

    def __len__(self):
        return len(self.audio_paths)

    def _parse_manifest(self):
        stage_manifest_path = os.path.join(self.manifest_path, f"{self.stage}.tsv")
        audio_paths = list()
        transcripts = list()
        with open(stage_manifest_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, transcript = line.split("\t")[0], line.split("\t")[1]
                transcript = transcript.replace("\n", "")
                audio_paths.append(os.path.join(self.dataset_path, audio_path))
                transcripts.append(transcript)
        return audio_paths, transcripts

    def _parse_transcript(self, tokens: str):
        transcript = list()
        transcript.extend(self.tokenizer.text2int(tokens))
        return transcript

    def _parse_audio(self, audio_path):
        signal, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            signal = torchaudio.functional.resample(signal, sr, self.sample_rate)

        signal, _ = self.speed_perturb(signal)

        feature = self.extract_feature(signal)  # [1, dim, time]
        if self.spec_aug is not None:
            feature = self.spec_aug(feature)  # [1, dim, time]

        # feature [1, dim, time] -> [time, dim]
        feature = feature.squeeze(0).transpose(1, 0)

        # TODO add noise_augment,spec_sub,spec_trim
        return feature
