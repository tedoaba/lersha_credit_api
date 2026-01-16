import numpy as np


def replace_inf(X):
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    return X
