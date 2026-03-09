import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y):
        y = np.array(y).reshape(-1)
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)
        if self.std_ == 0:
            self.std_ = 1
        return self

    def transform(self, y):
        y = np.array(y).reshape(-1)
        return (y - self.mean_) / self.std_

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_scaled):
        y_scaled = np.array(y_scaled).reshape(-1)
        return (y_scaled * self.std_) + self.mean_
