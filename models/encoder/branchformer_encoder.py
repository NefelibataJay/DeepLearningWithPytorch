import torch
from torch import nn, Tensor

from models.modules.attention import RelPositionMultiHeadedAttention
from models.modules.cgmlp import ConvolutionalGatingMLP
from models.modules.convolution import Conv2dSubsampling
from models.modules.embedding import RelPositionalEncoding
from models.modules.feed_forward import FeedForwardModule
from models.modules.mask import make_pad_mask


class BranchformerBlock(torch.nn.Module):
    def __init__(
            self,
            encoder_dim: int = 256,
            attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            cgmlp_linear_expansion_factor: int = 4,
            feed_forward_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            cgmlp_conv_dropout_rate: float = 0.1,
            cgmlp_conv_kernel_size: int = 31,
            dropout_rate: float = 0.1,
            use_linear_after_conv: bool = True,
            half_step_residual: bool = True,
            merge_type: str = "concat",
    ):
        super(BranchformerBlock, self).__init__()
        if half_step_residual:
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1

        self.dropout = torch.nn.Dropout(dropout_rate)
        # NOTE: Espnet uses a different order of modules
        # self.feed_forward_f = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate)

        self.atte_norm = nn.LayerNorm(encoder_dim)
        self.attn = RelPositionMultiHeadedAttention(
            attention_heads,
            encoder_dim,
            attention_dropout_rate,
        )

        self.cgmlp_norm = nn.LayerNorm(encoder_dim)
        self.cgmlp = ConvolutionalGatingMLP(
            encoder_dim,
            cgmlp_linear_expansion_factor * encoder_dim,
            cgmlp_conv_kernel_size,
            cgmlp_conv_dropout_rate,
            use_linear_after_conv,
        )

        self.merge_proj = torch.nn.Linear(encoder_dim + encoder_dim, encoder_dim)

        self.feed_forward_b = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_rate)

        self.final_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x_input, mask):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # x = x + self.feed_forward_f(x) * self.ff_scale
        # to two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.atte_norm(x1)
        if pos_emb is not None:
            x_att = self.attn(x1, x1, x1, pos_emb, mask)
        else:
            x_att = self.attn(x1, x1, x1, mask)
        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating MLP module
        x2 = self.cgmlp_norm(x2)
        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]
        x2 = self.dropout(x2)

        # Merge
        x_concat = torch.cat([x1, x2], dim=-1)
        x = x + self.dropout(self.merge_proj(x_concat))

        x = x + self.feed_forward_b(x) * self.ff_scale

        x = self.final_norm(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class BranchformerEncoder(torch.nn.Module):
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 256,
            attention_heads: int = 4,
            feed_forward_expansion_factor: int = 4,
            cgmlp_linear_expansion_factor: int = 4,
            cgmlp_conv_kernel_size: int = 31,
            num_layers: int = 12,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            feed_forward_dropout_rate: float = 0.1,
            cgmlp_conv_dropout_rate: float = 0.1,
            use_linear_after_conv: bool = True,
            half_step_residual: bool = True,
            merge_type: str = "concat",
    ):
        super().__init__()
        self.conv_subsample = Conv2dSubsampling(in_channels=1, output_dim=encoder_dim)
        self.input_projection = torch.nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.pos_enc = RelPositionalEncoding(encoder_dim, positional_dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)

        self.encoders = torch.nn.ModuleList([
            BranchformerBlock(
                encoder_dim=encoder_dim,
                attention_heads=attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                cgmlp_linear_expansion_factor=cgmlp_linear_expansion_factor,
                feed_forward_dropout_rate=feed_forward_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                cgmlp_conv_dropout_rate=cgmlp_conv_dropout_rate,
                cgmlp_conv_kernel_size=cgmlp_conv_kernel_size,
                dropout_rate=dropout_rate,
                use_linear_after_conv=use_linear_after_conv,
                half_step_residual=half_step_residual,
                merge_type=merge_type)
            for _ in range(num_layers)])

        self.after_norm = nn.LayerNorm(encoder_dim)

    def forward(self, inputs: Tensor, input_lengths: Tensor, ):
        outputs, outputs_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.pos_enc(self.input_projection(outputs))
        """ 
        We believe that Espnet made some errors in calculating the Mask length after the convolution
        So we use the following code to calculate the mask length
        """
        # (~make_pad_mask(input_lengths)[:, None, :]) == (~make_pad_mask(input_lengths).squeeze(1)
        masks = (~make_pad_mask(input_lengths)[:, None, :])[:, :, :-2:2][:, :, :-2:2].to(outputs[0].device)

        for layer in self.encoders:
            outputs, masks = layer(outputs, masks)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = self.after_norm(outputs)

        return outputs, outputs_lengths
