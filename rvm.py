import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.stats as sts

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split


class RVM(BaseEstimator, RegressorMixin):
    def __init__(self, n_features):
        self.n_features = n_features
        self.alpha = np.ones(n_features)
        self.a = np.ones(n_features)
        self.sigma_sq = 1
        self.A = np.diag(self.alpha)
        self.n_iter = 1000

    def fit(self, X, y, verbose=True):
        iters = 0
        self._rss_list = []
        self._n_samples = X.shape[0]

        while iters < self.n_iter and not self._check_condition():
            # alpha, возможно, успеют обновиться лишний раз.
            self._update_a(X, y)
            self._update_gamma()
            self._update_alpha()
            self._update_sigma_sq(X, y)
            
            iters += 1
        if verbose:
            self.rss_plot()

    def predict(self, X):
        return X @ self.a

    def _check_condition(self):
        min_cond = self.alpha < 10**(-6)
        max_cond = self.alpha > 10**12

        if np.any(min_cond):
            self.alpha[min_cond] = 0
            self.a[min_cond] = 100000
            return True
        if np.any(max_cond):
            self.alpha[max_cond] = np.inf
            self.a[max_cond] = 0
            return True
        return False

    def _update_a(self, X, y):
        self.A = np.diag(self.alpha)
        self.Sigma = inv(1 / self.sigma_sq * X.T @ X + self.A)
        self.a = 1 / self.sigma_sq * self.Sigma @ (X.T @ y)

    def _update_gamma(self):
        self.gamma = 1 - self.alpha * np.diag(self.Sigma)

    def _update_alpha(self):
        self.alpha = self.gamma / (self.a)**2

    def _update_sigma_sq(self, X, y):
        y_pred = self.predict(X)
        rss = np.dot(y - y_pred, y - y_pred)
        self._rss_list.append(rss)

        self.sigma_sq = rss / (self._n_samples - self.gamma.sum())

    def rss_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_title('Train.')
        ax.set_xlabel('Iters.')
        ax.set_ylabel('RSS.')
        
        _x = np.arange(len(self._rss_list))
        _y = self._rss_list
        ax.plot(_x, _y)

    def get_params(self):
        return {'n_features' : self.n_features}

    def set_params(self, *args):
        self.n_features = args[0]
