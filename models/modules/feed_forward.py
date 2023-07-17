# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor

from models.modules.modules import Linear


class PositionwiseFeedForward(nn.Module):
    def __init__(self, encoder_dim: int = 256, hidden_units: int = 2048, dropout_p: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim, eps=1e-5),
            nn.Linear(encoder_dim, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_units, encoder_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class FeedForwardModule(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 256,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim, eps=1e-5),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
