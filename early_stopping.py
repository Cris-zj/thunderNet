import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=4, verbose=False, delta=0.001,min_lr=1e-6,threshold=0.1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_lr=min_lr
        self.threshold=threshold

    def __call__(self, val_loss, solver):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, solver)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if(self.best_score<-self.threshold and solver.optimizer.param_groups[0]["lr"]>self.min_lr):
                    solver.set_lr()
                    self.counter=0
                else:
                    self.early_stop=True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, solver)
            self.counter = 0

    def save_checkpoint(self, val_loss, solver):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        solver.save()
        self.val_loss_min = val_loss