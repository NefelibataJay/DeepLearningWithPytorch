from typing import List
import torch
from omegaconf import DictConfig
from util.tokenize.tokenizer import Tokenizer


class EnglishCharTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig):
        super(EnglishCharTokenizer, self).__init__(configs)
        with open(self.word_dict_path, 'r', encoding='utf8') as dict_file:
            for line in dict_file:
                key, value = line.replace("\n", "").split('|')
                self.vocab[key] = int(value)
                self.id_dict[int(value)] = key

    def text2int(self, tokens: str) -> List[int]:
        label = []
        for ch in tokens:
            if ch in self.vocab:
                label.append(self.vocab[ch])
            elif '<unk>' in self.vocab:
                label.append(self.vocab['<unk>'])
        return label

    def int2text(self, t: torch.Tensor) -> str:
        sentence = str()
        for i in t:
            if i == self.eos_token:
                # i = eos
                break
            elif i == self.blank_token:
                # i = blank
                continue
            sentence += self.id_dict[int(i)]
        return sentence

    def __len__(self):
        return len(self.vocab)
