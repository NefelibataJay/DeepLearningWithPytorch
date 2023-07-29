import os

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from tensorboardX import SummaryWriter

from tool.common import remove_pad
from tool.tokenize.tokenizer import Tokenizer
from util.initialize import init_search


class ConformerCTCTrainer:
    def __init__(self, config: DictConfig, tokenizer: Tokenizer, model: torch.nn.Module, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, metric, device) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.device = device
        self.accum_grad = self.config.train_conf.accum_grad
        if self.config.train_conf.grad_clip is not None:
            self.grad_clip = self.config.train_conf.grad_clip
        self.logger = SummaryWriter(os.path.join(self.config.save_path, "log", self.config.model_name))
        self.search = init_search(self.config)
        # TODO add early stop

    def train(self, train_dataloader, valid_dataloader):
        # first validate
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
                loss /= self.accum_grad
                loss.backward()

                if (idx + 1) % self.accum_grad == 0 or (idx + 1) == len(train_dataloader):
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip, norm_type=2)
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
        self.model.eval()
        print("=========================Eval=========================")
        valid_loss = 0
        bar = tqdm(enumerate(valid_dataloader), desc=f"Training Eval")
        for idx, batch in bar:
            inputs, input_lengths, targets, target_lengths = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            input_lengths = input_lengths.to(self.device)
            target_lengths = target_lengths.to(self.device)
            result = self.model(inputs, input_lengths, targets, target_lengths)
            loss = result["loss"]
            valid_loss += loss.item()
            bar.set_postfix(loss='{:.4f}'.format(loss.item()))

        valid_loss /= len(valid_dataloader)
        self.logger.add_scalar("valid_loss", valid_loss, epoch)
        bar.set_postfix(val_loss='{:.4f}'.format(valid_loss))
        print("valid_loss:", valid_loss)

    @torch.no_grad()
    def test(self, test_dataloader):
        self.model.eval()
        print("=========================Test=========================")
        test_acc = 0
        bar = tqdm(enumerate(test_dataloader), desc=f"Test")
        for idx, batch in bar:
            inputs, input_lengths, targets, target_lengths = batch
            inputs = inputs.to(self.device)
            input_lengths = input_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            result = self.model(inputs, input_lengths, targets, target_lengths)

            hyps, _ = self.search.ctc_greedy_search(self.model, result["encoder_outputs"], result["output_lengths"])
            # decoding
            predictions = [self.tokenizer.int2text(sent) for sent in hyps]
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

    def save_model(self, epoch):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.save_path, "checkpoints", f"{self.config.model_name}_{epoch}.pt"))
