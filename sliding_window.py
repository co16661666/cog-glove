import numpy as np

class SlidingWindow:
    def __init__(self, window_length, threshold):
        self.window_length = window_length
        self.threshold = threshold
        self.window = np.full(window_length, np.nan)
        self.initial_count = 0

    def is_outlier(self, new_value):
        if self.initial_count < self.window_length:
            self.initial_count += 1
            return False  # Window still empty
        
        mean = np.nanmean(self.window)
        std = np.nanstd(self.window)

        self.update(new_value)

        z_score = abs(new_value - mean) / std
        return z_score > self.threshold

    def update(self, new_value):
        self.window = np.roll(self.window, -1)
        self.window[-1] = new_value

        if self.initial_count < self.window_length:
            self.initial_count += 1

    