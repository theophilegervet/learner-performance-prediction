class TimeWindowQueue:
    """A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.

    From JJ's KTM repository: https://github.com/jilljenn/ktm.
    """
    def __init__(self, window_lengths):
        self.queue = []
        self.window_lengths = window_lengths
        self.cursors = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        return [len(self.queue)] + [len(self.queue) - cursor for cursor in self.cursors]

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1
