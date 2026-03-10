import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        data = self._check_ndim(np.array(data))

        self.mean_ = np.mean(data, axis=0)  # sütun bazlı
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, data):
        data = self._check_ndim(np.array(data))
        return (data - self.mean_) / self.std_
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        data = self._check_ndim(np.array(data))
        return (data * self.std_) + self.mean_
    
    def _check_ndim(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data
