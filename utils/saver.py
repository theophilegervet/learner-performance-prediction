import os
import torch
import numpy as np


class Saver:
    """Saving pytorch model.
    """

    def __init__(self, savedir, filename, patience=10):
        """
        Arguments:
            savedir (str): Save directory.
            filename (str): Save file name.
            patience (int): How long to wait after last time validation loss improved.
        """
        self.path = os.path.join(savedir, filename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        self.patience = patience
        self.score_max = -np.Inf
        self.counter = 0
        self.best_epoch = -1

    def save(self, score, network, epoch=None):
        """
        Arguments:
            score (float): Score to maximize.
            network (torch.nn.Module): Network to save if validation loss decreases.
            epoch: Epoch number
        """
        if score <= self.score_max:
            self.counter += 1
        else:
            self.score_max = score
            if epoch is not None:
                self.best_epoch = epoch
            torch.save(network, self.path)
            self.counter = 0

        stop = self.counter >= self.patience
        return stop

    def load(self):
        return torch.load(self.path)
