import os
import shutil
import random
import torch
from tensorboardX import SummaryWriter


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class Logger:
    """Logging with TensorboardX.
    """
    def __init__(self, logdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass

        self.writer = SummaryWriter(logdir)

    def log_scalar(self, tag, value, step):
        """Log scalar value.
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, array, step):
        """Log histogram of numpy array of values.
        """
        self.writer.add_histogram(tag, array, step)
        
    def close(self):
        self.writer.close()


class Metrics:
    """Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def store(self, new_metrics):
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
                self.counts[key] += 1
            else:
                self.metrics[key] = new_metrics[key]
                self.counts[key] = 1

    def average(self):
        average = {k: v / self.counts[k] for k, v in self.metrics.items()}
        self.metrics, self.counts = {}, {}
        return average