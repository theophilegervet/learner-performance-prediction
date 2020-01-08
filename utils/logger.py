import os
import shutil
from tensorboardX import SummaryWriter


class Logger:
    """Logging with TensorboardX. """
    def __init__(self, logdir, verbose=True):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass

        self.verbose = verbose
        self.writer = SummaryWriter(logdir)

    def log_histograms(self, dic, step):
        """Log dictionary of tensors as histograms. """
        for k, v in dic.items():
            self.writer.add_histogram(k, v, step)

    def log_scalars(self, dic, step):
        """Log dictionary of scalar values. """
        for k, v in dic.items():
            self.writer.add_scalar(k, v, step)

        if self.verbose:
            print(f"Step {step}, {dic}")

    def close(self):
        self.writer.close()