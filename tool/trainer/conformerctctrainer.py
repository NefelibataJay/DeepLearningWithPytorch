import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from util.tokenize.tokenizer import Tokenizer


class ConformerCTCTrainer:
    def __init__(self, config: DictConfig, tokenizer: Tokenizer, model: torch.nn.Module, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, criterion: torch.nn, metric, device) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def train(self, train_dataloader, valid_dataloader):
        self.model.to(self.device)
        for epoch in range(self.config.train_conf.max_epoch):
            print("Epoch:", epoch)
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_dataloader):
                self.optimizer.zero_grad()
                inputs, input_lengths, targets, target_lengths = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)

                encoder_outputs, output_lengths, logits = self.model(inputs, input_lengths)

                loss = self.criterion(
                    hs_pad=logits,
                    ys_pad=targets,
                    h_lens=output_lengths,
                    ys_lens=target_lengths,
                )
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            print("train_loss:", train_loss, "train_lr", self.scheduler.get_lr())
            self.scheduler.step(train_loss)
            if epoch % self.config.train_conf.valid_interval == 0:
                self.validate(valid_dataloader)
            if epoch % self.config.train_conf.save_interval == 0:
                self.save_model()

    def validate(self, valid_dataloader):
        self.model.eval()
        valid_loss = 0
        valid_acc = 0
        for batch in tqdm(valid_dataloader):
            inputs, input_lengths, targets, target_lengths = batch
        valid_loss /= len(valid_dataloader)
        valid_acc /= len(valid_dataloader)
        print("valid_loss:", valid_loss, "valid_acc:", valid_acc)
        return valid_acc

    def save_model(self, epoch):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, f"{self.config.model_name}_{epoch}.pt"))
