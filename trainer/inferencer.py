import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from tool.common import remove_pad
from tool.tokenize.tokenizer import Tokenizer
from util.initialize import init_search, init_metric


class SpeechToText:
    def __init__(self,
                 config: DictConfig,
                 tokenizer: Tokenizer,
                 model: torch.nn.Module,
                 device):
        self.config = config
        self.logger = SummaryWriter(os.path.join(self.config.save_path, "log", self.config.model_name))
        self.tokenizer = tokenizer
        self.model = model
        self.metric = init_metric(config)
        self.decode = init_search(config)
        self.device = device

        self.search = init_search(config)

    @torch.no_grad()
    def recognition(self, test_dataloader):
        self.model.to(self.device).eval()
        print("=========================Test=========================")
        test_acc = 0
        bar = tqdm(enumerate(test_dataloader), desc=f"Test")
        for idx, batch in bar:
            inputs, input_lengths, targets, target_lengths = batch
            inputs = inputs.to(self.device)
            input_lengths = input_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # decoding
            best_hyps, _ = self.search(inputs, input_lengths, self.model)

            predictions = [self.tokenizer.int2text(sent) for sent in best_hyps]
            targets = [self.tokenizer.int2text(sent) for sent in targets]
            self.metric(predictions, targets)
            char_error_rate = self.metric.compute() * 100
            test_acc += char_error_rate
            bar.set_postfix(cer='{:.4f}'.format(char_error_rate))

        test_acc /= len(test_dataloader)
        self.logger.add_scalar("test_acc", test_acc)
        print("test_acc:", test_acc)
