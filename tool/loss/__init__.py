from tool.loss.ctc import CTC
from tool.loss.label_smoothing_loss import LabelSmoothingLoss
from torch.nn import CrossEntropyLoss

REGISTER_LOSS = {
    "ctc": CTC,
    "label_smoothing_loss": LabelSmoothingLoss,
    "cross_entropy_loss": CrossEntropyLoss,
}
