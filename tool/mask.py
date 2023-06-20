import torch
from torch import Tensor


def get_attn_pad_mask(inputs, input_lengths, expand_length):
    """mask position is set to 1"""

    def get_transformer_non_pad_mask(inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """Padding position is set to 0, either use input_lengths or pad_id"""
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())  # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")

        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

        return non_pad_mask

    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        max_len:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    return ~make_pad_mask(lengths)
