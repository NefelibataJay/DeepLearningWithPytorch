from tool.optimizer.warmup_lr import WarmupLR
from tool.optimizer.gradual_warmup_lr import GradualWarmupScheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, \
    CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR, CyclicLR
from torch.optim import ASGD, SGD, Adagrad, Adam, Adamax, AdamW

REGISTER_SCHEDULER = {
    "warmup_lr": WarmupLR,
    "gradual_warmup_lr": GradualWarmupScheduler,
    "lambda_lr": LambdaLR,
    "step_lr": StepLR,
    "multi_step_lr": MultiStepLR,
    "exponential_lr": ExponentialLR,
    "cosine_annealing_lr": CosineAnnealingLR,
    "cosine_annealing_warm_restarts": CosineAnnealingWarmRestarts,
    "one_cycle_lr": OneCycleLR,
    "reduce_lr_on_plateau": ReduceLROnPlateau,
    "cyclic_lr": CyclicLR,
}

REGISTER_OPTIMIZER = {
    "adam": Adam,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "adamw": AdamW,
    "sgd": SGD,
    "asgd": ASGD,
}
