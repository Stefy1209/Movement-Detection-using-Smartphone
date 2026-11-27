import numpy as np

class SVM:
    def __init__(self, C: float, seed: int = 0):
        self._C = C
        self._rng = np.random.default_rng(seed)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        K = len(self.classes_)

        if 2 > K:
            raise ValueError("y must contain at least 2 classes.")

        self.W_ = np.zeros((K, X.shape[1]))

        for k, cls in enumerate(self.classes_):
            y_k = np.where(y == cls, 1, -1)
            self.W_[k] = self._binary_fit(X, y = y_k)
        
        return self

    def _binary_fit(self, X: np.ndarray, y: np.ndarray, T: int = 1000):
        n_samples, n_features = X.shape
        lambda_ = 1 / (self._C * n_samples)
        w_ = np.zeros(n_features)

        for t in range(1, T+1):
            i = self._rng.integers(n_samples)
            niu_ = 1 / (lambda_ * t)

            if y[i] * (w_ @ X[i]) < 1:
                w_ = (1 - niu_ * lambda_) * w_ + niu_ * y[i] * X[i]

            else:
                w_ = (1 - niu_ * lambda_) * w_

        return w_

    def decision_function(self, X):
        X = np.asarray(X)
        return X @ self.W_.T

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]