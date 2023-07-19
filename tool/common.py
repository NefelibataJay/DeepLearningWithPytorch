from typing import List, Tuple
import torch


def remove_duplicates_and_blank(hyp: List[int], blank=4) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def replace_duplicates_with_blank(hyp: List[int], blank=4) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(hyp) and hyp[cur] == hyp[prev] and hyp[cur] != 0:
            new_hyp.append(blank)
            cur += 1
    return new_hyp


def remove_pad(ys_pad: torch.Tensor, pad=0):
    ys = []
    for y in ys_pad:
        if y != pad:
            ys.append(y)
    return ys


def add_sos(y_pad, sos: int = 1, pad: int = 0):
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=y_pad.device)
    ys = [y[y != pad] for y in y_pad]
    ys_sos = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_sos = torch.nn.utils.rnn.pad_sequence(ys_sos, batch_first=True, padding_value=0)
    return ys_sos


def add_eos(y_pad, eos: int = 2, pad: int = 0):
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=y_pad.device)
    ys = [y[y != pad] for y in y_pad]
    ys_eos = [torch.cat([y, _eos], dim=0) for y in ys]
    ys_eos = torch.nn.utils.rnn.pad_sequence(ys_eos, batch_first=True, padding_value=0)
    return ys_eos


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int, pad: int):
    """Add <sos> and <eos> labels."""
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != pad] for y in ys_pad]
    ys_eos_sos = [torch.cat([_sos, y, _eos], dim=0) for y in ys]
    ys_eos_sos = torch.nn.utils.rnn.pad_sequence(ys_eos_sos, batch_first=True, padding_value=0)
    return ys_eos_sos


def add_blank(ys_pad: torch.Tensor, blank: int,
              ignore_id: int) -> torch.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ignore_id: padding index
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)

        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


if __name__ == "__main__":
    blank = 3
    ys_pad = torch.tensor([[1, 2, 3, 4, 5],
                           [4, 5, 6, 0, 0],
                           [7, 8, 9, 0, 0]], dtype=torch.int32)
    print(add_blank(ys_pad, blank, ignore_id=0))
    rnnt_text = torch.where(ys_pad == 0, -1, ys_pad).to(torch.int32)
    print(rnnt_text)
