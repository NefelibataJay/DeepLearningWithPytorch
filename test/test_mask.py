import torch

from models.modules.mask import make_pad_mask


def make_pad_mask1(lengths, xs=None, length_dim=-1, maxlen=None):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def test_mask():
    lengths = torch.tensor([50, 60, 70])
    # print(make_pad_mask(lengths))
    # print(~make_pad_mask(lengths))
    # print(make_pad_mask1(lengths))
    # print(~make_pad_mask1(lengths))
    masks = (~make_pad_mask(lengths)[:, None, :])
    print(masks.shape)
    masks = masks[:, :, :-2:2][:, :, :-2:2]
    print(masks.shape)
    print(masks)
    o_len = lengths >> 2
    o_len = o_len - 1
    print(o_len)
    print(masks.squeeze(1).sum(1))
    print((~make_pad_mask(o_len)[:, None, :]).squeeze(1).sum(1))

