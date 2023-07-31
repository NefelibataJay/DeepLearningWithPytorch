import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from tensorboardX import SummaryWriter

from tool.tokenize.tokenizer import Tokenizer


class Trainer:
    def __init__(self, config: DictConfig,
                 tokenizer: Tokenizer,
                 model: torch.nn.Module, frontend,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 metric, device) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.frontend = frontend  # TODO: add frontend
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.device = device
        self.logger = SummaryWriter(os.path.join(self.config.save_path, "log", self.config.model_name))

        if self.config.train_conf.grad_clip is not None:
            self.grad_clip = self.config.train_conf.grad_clip

    def train(self, train_dataloader, valid_dataloader):
        self.validate(valid_dataloader, -1)
        self.model.to(self.device)
        print("=========================Start Training=========================")
        for epoch in range(self.config.train_conf.max_epoch + 1):
            self.model.train()
            train_loss = 0
            self.optimizer.zero_grad()

            bar = tqdm(enumerate(train_dataloader), desc=f"Training Epoch:{epoch}")
            for idx, batch in bar:
                inputs, input_lengths, targets, target_lengths = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)

                result = self.model(inputs, input_lengths, targets, target_lengths)
                loss = result["loss"]

                loss /= self.config.train_conf.accum_grad
                loss.backward()

                if (idx + 1) % self.config.train_conf.accum_grad == 0 or (idx + 1) == len(train_dataloader):
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.config.train_conf.grad_clip, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                bar.set_postfix(loss='{:.4f}'.format(loss.item()))
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            self.logger.add_scalar("train_loss", train_loss, epoch)
            self.logger.add_scalar("train_lr", self.scheduler.get_last_lr(), epoch)
            self.scheduler.step()

            if epoch % self.config.train_conf.valid_interval == 0 or epoch == self.config.train_conf.max_epoch:
                self.validate(valid_dataloader, epoch)
            if epoch % self.config.train_conf.save_interval == 0 or epoch == self.config.train_conf.max_epoch:
                self.save_model(epoch)

    @torch.no_grad()
    def validate(self, valid_dataloader, epoch):
        pass

    def save_model(self, epoch):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.save_path, "checkpoints", f"{self.config.model_name}_{epoch}.pt"))
