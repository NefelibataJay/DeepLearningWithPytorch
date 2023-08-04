import torch
from torch import optim

from tool.optimizer import WarmupLR, GradualWarmupScheduler
from tool.optimizer.warmup_step_lr import WarmupStepLR
from torch.optim.lr_scheduler import StepLR


def test_lr():
    model = torch.nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupLR(optimizer, warmup_steps=5)
    # scheduler = WarmupStepLR(optimizer, warmup_steps=5,steps_per_epoch=10,step_size=2)
    # scheduler = GradualWarmupScheduler(optimizer,total_epoch=10,multiplier=2, after_scheduler=StepLR(optimizer, step_size=2))

    for i in range(1,30):
        scheduler.step()
        print(f"step:{i},{scheduler.get_lr()}")
