import torch
from torch import nn

from models.modules.attention import MultiHeadedAttention
from models.modules.feed_forward import PositionwiseFeedForward
from models.modules.mask import make_pad_mask, subsequent_mask


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self,
                 attention_dim: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.1,
                 src_attention_dropout_rate: float = 0.1,
                 ):
        super().__init__()

        self.norm1 = nn.LayerNorm(attention_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(attention_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)

        self.self_attn = MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate)
        self.src_attn = MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate)
        self.ff = PositionwiseFeedForward(attention_dim, linear_units, dropout_rate)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """
        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, attention_dim).
            tgt_mask (torch.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory
                (#batch, maxlen_in, attention_dim).
            memory_mask (torch.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (torch.Tensor): cached tensors.
                (#batch, maxlen_out - 1, attention_dim).
        Returns:
            torch.Tensor: Output tensor (#batch, maxlen_out, attention_dim).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, attention_dim).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt

        tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask)[0])

        residual = x
        x = residual + self.dropout(self.feed_forward(x))

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 attention_dim: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, attention_dim)
        self.pe = None

        self.before_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)

        self.num_blocks = num_layers
        self.decoders = torch.nn.ModuleList([
            TransformerDecoderBlock(attention_dim, attention_heads, linear_units, dropout_rate,
                                    self_attention_dropout_rate, src_attention_dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor, ys_in_pad: torch.Tensor,
                ys_in_lens: torch.Tensor, ):
        """
         Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                olens: (batch, )
        """
        tgt = ys_in_pad
        max_len = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, max_len).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        x = self.before_norm(x)
        x = self.output_layer(x)
        olens = tgt_mask.sum(1)
        return x, olens
