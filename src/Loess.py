import numpy as np
import time
import math

from sklearn.preprocessing import PolynomialFeatures

def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y


class Loess(object):

    @staticmethod
    def normalize_data(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_data(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_data(yy)
    
    @staticmethod
    def get_weights(distances, min_range):
        max_distance = distances[min_range, np.arange(0, distances.shape[1])][-1]
        weights = tricubic(distances[min_range, np.arange(0, distances.shape[1])] / max_distance)
        return weights

    @staticmethod
    def get_min_range(distances, window):
        return np.argsort(distances, axis=0)[:window]

    def normalize_x(self, xx):
        return (xx - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def weighted_pseudo_inverse(self, X, W):
        Xt_W = np.transpose(X) @ W
        return np.linalg.inv(Xt_W @ X) @ Xt_W

    def estimate(self, x : np.array, window : int, degree : int = 1):
        """
        Estimates f(x) based on observations given set of inputs.

        Parameters
        ----------
            x : np.array
                input point to predict
            window : int
                neighborhood size
            degree : int
                regression type (1 = linear, 2 = quadratic)
        """

        normalized_x = self.normalize_x(x)

        distances = np.abs(self.n_xx - normalized_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)
        
        pf = PolynomialFeatures(degree)

        xp = pf.fit_transform([normalized_x])
        X1 = self.n_xx[min_range, np.arange(0, self.n_xx.shape[1])]
        X1 = pf.fit_transform(X1)

        print(X1.T)
        print(weights)

        W = np.diag(weights)

        # (X^TWX)^-1 X^TW
        WPinv = self.weighted_pseudo_inverse(X1, weights)
        
        Y = self.n_yy[min_range]

        beta = WPinv @ Y

        return self.denormalize_y(beta @ xp)
