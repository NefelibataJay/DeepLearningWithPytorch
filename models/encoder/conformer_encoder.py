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

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from models.modules.embedding import RelPositionalEncoding
from models.modules.feed_forward import FeedForwardModule
from models.modules.attention import RelPositionMultiHeadedAttention
from models.modules.convolution import (
    Conv2dSubsampling, ConvolutionModule,
)
from models.modules.mask import make_pad_mask
from models.modules.modules import (
    ResidualConnectionModule,
    Linear,
)


class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 256,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1
        self.dropout = nn.Dropout(dropout_p)

        self.f1 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.norm_mha = nn.LayerNorm(encoder_dim)
        self.self_attn = RelPositionMultiHeadedAttention(num_attention_heads, encoder_dim, attention_dropout_p)

        self.norm_conv = nn.LayerNorm(encoder_dim)
        self.conv_module = ConvolutionModule(encoder_dim, conv_kernel_size)

        self.f2 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.norm_final = nn.LayerNorm(encoder_dim)

    def forward(self, x_input, mask, cache=None):
        """
            Inputs: inputs
                - **inputs** (batch, time, dim): Tensor containing input vector
                - mask (batch, 1, time)

            Returns: outputs
                - **outputs** (batch, time, dim): Tensor produces by conformer block.
            """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # first feed forward
        x = x + self.ff_scale * self.f1(x)

        # attention
        residual = x
        x = self.norm_mha(x)

        if pos_emb is not None:
            x_att = self.self_attn(x, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x_att)

        # convolution
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x))

        # last feed forward
        x = x + self.ff_scale * self.f2(x)

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask


class ConformerEncoder(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        dropout_p (float, optional): Probability of conformer dropout
        conv_kernel_size (int or tuple, optional): Size of the convolution kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 256,
            num_layers: int = 12,
            num_attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2dSubsampling(in_channels=1, output_dim=encoder_dim)

        self.input_projection = Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.pos_encoding = RelPositionalEncoding(encoder_dim, input_dropout_p)

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            dropout_p=dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs, pos_emb = self.pos_encoding(self.input_projection(outputs))

        # NOTE: the ture lengths is output_lengths, but output_lengths was negative 1
        # output_lengths -= 1
        # so we use input_lengths instead and deal with espnet
        masks = (~make_pad_mask(input_lengths)[:, None, :])[:, :, :-2:2][:, :, :-2:2].to(outputs.device)
        outputs = (outputs, pos_emb)

        for layer in self.layers:
            outputs, masks = layer(outputs, masks)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        return outputs, output_lengths
