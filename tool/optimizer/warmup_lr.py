from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_epoch: Union[int, float] = 20,
            last_epoch: int = -1,
    ):
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_epoch ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_epoch ** -1.5)
            for lr in self.base_lrs
        ]
