import torch


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> float:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B , Lmax, Class num).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.
    Returns:
        float: Accuracy value (0.0 -- 1.0).
    """
    pad_pred = pad_outputs.argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)
