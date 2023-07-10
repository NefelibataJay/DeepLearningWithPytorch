""" Espnet CTC module """

import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(
            self,
            blank_id: int = 0,
            reduction: str = 'sum',
            zero_infinity: bool = True,
    ):
        """ Construct CTC module """
        super().__init__()
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, hs_pad: torch.Tensor, h_lens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            h_lens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        ys_hat = hs_pad.transpose(0, 1).log_softmax(dim=-1)
        loss = self.ctc_loss(ys_hat, ys_pad, h_lens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss
