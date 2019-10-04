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