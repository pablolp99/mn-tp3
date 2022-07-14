import numpy as np
import time
import math

from sklearn.preprocessing import PolynomialFeatures

def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= 0) & (x < 1)
    y[idx] = np.power(1.0 - np.power(x[idx], 3), 3)
    return y

def bi_square(xx, **kwargs):
    """
    The bi-square weight function calculated over values of xx
    Parameters
    ----------
    xx: float array
    """
    ans = np.zeros(xx.shape)
    idx = np.where(xx < 1)
    ans[idx] = (1 - xx[idx] ** 2) ** 2
    return ans

def unweighted(x):
    return np.ones_like(x)

kernel_map = {
    "bisquare": bi_square,
    "tricubic": tricubic,
    "unweighted": unweighted,
}

class Loess(object):

    def __init__(self, xx, yy):
        self.n_xx = xx
        self.n_yy = yy
    
    @staticmethod
    def get_weights(distances, min_range, kernel):
        max_distance = np.max(distances[min_range])
        weights = kernel(distances[min_range] / max_distance)
        return weights

    @staticmethod
    def get_min_range(distances, window):
        return np.argsort(distances, axis=0)[:window]

    def weighted_pseudo_inverse(self, X, W):
        Xt_W = np.dot(X.T, W)
        temp = np.linalg.pinv(np.dot(Xt_W, X))
        return np.dot(temp, Xt_W)

    def estimate(self, x : np.array, kernel : str, window : int, degree : int = 1):
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

        distances = np.array([np.linalg.norm(_x - x) for _x in self.n_xx])
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range, kernel_map[kernel])
        
        # Extending matrices depending on degree
        pf = PolynomialFeatures(degree)

        xp = pf.fit_transform([x])[0]
        X1 = self.n_xx[min_range]
        if len(X1.shape) == 1:
            X1 = pf.fit_transform(X1.reshape(-1, 1))
        else:
            X1 = pf.fit_transform(X1)

        # Weight matrix
        W = np.diag(weights)

        # Weighted Pseudo Inverse
        WPinv = self.weighted_pseudo_inverse(X1, W)
        
        y = self.n_yy[min_range]

        beta = np.dot(WPinv, y)

        return np.dot(beta, xp)
