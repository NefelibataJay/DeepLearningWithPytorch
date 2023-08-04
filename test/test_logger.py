from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim.lr_scheduler import StepLR


def test_logger():
    logger = SummaryWriter("./log")
