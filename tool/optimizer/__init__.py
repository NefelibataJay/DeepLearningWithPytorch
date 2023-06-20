from tool.optimizer.warmup_lr_scheduler import WarmupLR
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, LinearLR, SequentialLR, CyclicLR, \
    CosineAnnealingLR
from torch.optim import ASGD, SGD, Adagrad, Adam, Adamax, AdamW

REGISTER_SCHEDULER = {
    "warmup_lr": WarmupLR,
    "lambda_lr": LambdaLR,
    "step_lr": StepLR,
    "multi_step_lr": MultiStepLR,
    "exponential_lr": ExponentialLR,
    "linear_lr": LinearLR,
    "sequential_lr": SequentialLR,
    "cyclic_lr": CyclicLR,
    "cosine_annealing_lr": CosineAnnealingLR,
}

REGISTER_OPTIMIZER = {
    "adam": Adam,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "adamw": AdamW,
    "sgd": SGD,
    "asgd": ASGD,
}
