import os

import torchaudio
from omegaconf import DictConfig

from datasets import ASRDataset
from tool.tokenize.tokenizer import Tokenizer


class AccentDataset(ASRDataset):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer, stage):
        super(AccentDataset, self).__init__(configs, tokenizer, stage)
        # TODO add speaker list (file) and accent list (file)

        self.speaker_dict = None
        self.accent_dict = None

    def __getitem__(self, idx):
        audio_path, transcript, accent, speaker = self.data_dict[idx]
        speech_feature = self._parse_audio(audio_path)  # [time, dim]
        transcript = self._parse_transcript(transcript)
        input_lengths = speech_feature.size(0)  # time
        target_lengths = len(transcript)
        accent_id = self.accent_dict[accent]
        speaker_id = self.speaker_dict[speaker]

        return speech_feature, input_lengths, transcript, target_lengths, accent_id, speaker_id

    def _parse_transcript(self, tokens: str):
        transcript = list()
        transcript.extend(self.tokenizer.text2int(tokens))
        return transcript

    def _parse_speaker(self):
        pass

    def _parse_accent(self):
        pass

    def _parse_manifest(self):
        """
        data_dict[0] = [audio_path, transcript,accent, speaker]

        manifest_file must be
        column1 : audio_path
        column2 : transcript  (No punctuation)
        column3 : accent
        column4 : speaker
        ....
        """
        stage_manifest_path = os.path.join(self.manifest_path, f"{self.stage}.tsv")
        data_dict = dict()

        with open(stage_manifest_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = line.replace("\n", "")
                audio_path, transcript, accent, speaker = line.split("\t")[0], line.split("\t")[1], \
                    line.split("\t")[2], line.split("\t")[3]
                transcript = transcript.replace("\n", "")
                data_dict[idx] = [os.path.join(self.dataset_path, audio_path), transcript, accent, speaker]
        return data_dict
