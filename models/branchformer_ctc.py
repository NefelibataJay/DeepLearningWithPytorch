import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear


class BranchformerCTC(torch.nn.Module):
    def __init__(self, configs: DictConfig):
        super(BranchformerCTC, self).__init__()
