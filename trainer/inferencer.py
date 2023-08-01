import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from tensorboardX import SummaryWriter

from tool.common import remove_pad
from tool.tokenize.tokenizer import Tokenizer
from util.initialize import init_search


class SpeechToText:
    def __init__(self,
                 config: DictConfig,
                 tokenizer: Tokenizer,
                 model: torch.nn.Module,
                 metric, device):
        self.config = config
        self.logger = SummaryWriter(os.path.join(self.config.save_path, "log", self.config.model_name))
        self.tokenizer = tokenizer
        self.model = model
        self.metric = metric
        self.decode = init_search(config)
        self.device = device

        weights = dict(
            decoder=1.0 - self.config.weight_conf.ctc_weight,
            ctc=self.config.weight_conf.ctc_weight,
            length_bonus=self.config.weight_conf.penalty,
        )

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
            targets = [self.tokenizer.int2text(remove_pad(sent)) for sent in targets]
            list_cer = []
            for i, j in zip(predictions, targets):
                list_cer.append(self.metric(i, j))
            char_error_rate = torch.mean(torch.tensor(list_cer)) * 100
            test_acc += char_error_rate
            bar.set_postfix(acc='{:.4f}'.format(test_acc))

        test_acc /= len(test_dataloader)
        self.logger.add_scalar("test_acc", test_acc)
        print("test_acc:", test_acc)
