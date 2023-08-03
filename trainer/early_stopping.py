import torch


class EarlyStopping:
    def __init(self,patience=7, verbose=False, save_path='../output/checkpoint/model_name.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.loss_dict = []

    def __call__(self, model):
        pass


