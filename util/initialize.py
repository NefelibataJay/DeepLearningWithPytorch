import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random


def init_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_model(model_name):
    pass

def init_optimizer(optimizer_name):
    pass

def init_scheduler(scheduler_name):
    pass

def init_loss(loss_name):
    pass

def init_metric(metric_name):
    pass


