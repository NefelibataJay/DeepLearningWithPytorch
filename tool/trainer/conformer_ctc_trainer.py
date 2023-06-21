import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tool.tokenize.tokenizer import Tokenizer


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
        self.accum_grad = self.config.train_conf.accum_grad
        self.grad_clip = self.config.train_conf.grad_clip
        self.logger = SummaryWriter(os.path.join(self.config.save_path, "log", self.config.model_name))

    def train(self, train_dataloader, valid_dataloader):
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

                encoder_outputs, output_lengths, logits = self.model(inputs, input_lengths)

                loss = self.criterion(
                    hs_pad=logits,
                    ys_pad=targets,
                    h_lens=output_lengths,
                    ys_lens=target_lengths,
                )

                loss /= self.accum_grad
                loss.backward()

                if (idx + 1) % self.accum_grad == 0 or (idx + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip, norm_type=2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            self.logger.add_scalar("train_loss", train_loss, epoch)
            self.logger.add_scalar("train_lr", self.scheduler.get_last_lr(), epoch)

            bar.set_postfix(info=f"train_loss:{train_loss}, train_lr:{self.scheduler.get_lr()}")

            self.scheduler.step()

            if epoch % self.config.train_conf.valid_interval == 0:
                self.validate(valid_dataloader, epoch)
            if epoch % self.config.train_conf.save_interval == 0:
                self.save_model(epoch)

    @torch.no_grad()
    def validate(self, valid_dataloader, epoch):
        self.model.eval()
        print("=========================Eval=========================")
        valid_loss = 0
        valid_acc = 0
        for batch in tqdm(valid_dataloader):
            inputs, input_lengths, targets, target_lengths = batch
        valid_loss /= len(valid_dataloader)
        valid_acc /= len(valid_dataloader)
        self.logger.add_scalar("valid_loss", self.scheduler.get_lr(), epoch)
        print("valid_loss:", valid_loss, "valid_acc:", valid_acc)

    def save_model(self, epoch):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.save_path, "checkpoints", f"{self.config.model_name}_{epoch}.pt"))
