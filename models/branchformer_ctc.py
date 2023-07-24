import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from models.encoder.branchformer_encoder import BranchformerEncoder


class BranchformerCTC(torch.nn.Module):
    def __init__(self,configs: DictConfig):
        super(BranchformerCTC, self).__init__()

