import os
from omegaconf import DictConfig


class Tokenizer:
    def __init__(self, configs: DictConfig):
        self.sos_token = configs.tokenizer.sos_id
        self.eos_token = configs.tokenizer.eos_id
        self.pad_token = configs.tokenizer.pad_id
        self.blank_token = configs.tokenizer.blank_id
        self.word_dict_path = os.path.join(configs.dataset.manifest_path, 'vocab.txt')

        self.vocab = {}
        self.id_dict = {}

    def text2int(self, tokens):
        pass

    def int2text(self, t):
        pass

    def __len__(self):
        return len(self.vocab)
