import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        data = self._check_ndim(np.array(data))

        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        # max == min ise (sabit sütun) bölme hatasını önle
        self.max_[self.max_ == self.min_] = self.min_[self.max_ == self.min_] + 1
        return self

    def transform(self, data):
        data = self._check_ndim(np.array(data))
        return (data - self.min_) / (self.max_ - self.min_)
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        data = self._check_ndim(np.array(data))
        return (data * (self.max_ - self.min_)) + self.min_
    
    def _check_ndim(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data
