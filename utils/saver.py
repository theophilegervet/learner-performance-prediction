import os
import torch


class Saver:
    """Saving pytorch model.
    """
    def __init__(self, savedir, filename):
        self.path = os.path.join(savedir, filename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    def save(self, model):
        torch.save(model, self.path)
