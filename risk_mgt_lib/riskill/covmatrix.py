import numpy as np
import pandas as pd


def pop_weights(w: np.ndarray, lamda: float) -> np.ndarray:
    n = w.size
    for i in range(n):
        w[i] = (1 - lamda) * lamda ** (n - i - 1)
    total = w.sum()
    w = w / total
    return w


def pop_ew_cov(data: pd.DataFrame, lamda: float) -> np.ndarray:
    w = np.zeros(data.shape[0])
    w = pop_weights(w, lamda)
    for col in data.columns:
        data[col] = data[col] - data[col].mean()
    result = np.array(data.multiply(w, axis=0).T@data)
    return result


def pop_cov(data: pd.DataFrame) -> np.ndarray:
    for col in data.columns:
        data[col] = data[col] - data[col].mean()
    result = np.array(data.T @ data) / data.shape[0]
    return result


def pop_cor_ew_var(data: pd.DataFrame, lamda: float) -> np.ndarray:
    ew_cov = pop_ew_cov(data, lamda)
    ew_std_div = np.diag(np.sqrt(np.diagonal(ew_cov)))
    cov = pop_cov(data)
    std_div1 = np.diag(np.sqrt(np.diagonal(cov))**(-1))
    cor = std_div1 @ cov @ std_div1
    cor_ew_var = ew_std_div @ cor @ ew_std_div
    return cor_ew_var


def pop_ew_cor_var(data: pd.DataFrame, lamda: float) -> np.ndarray:
    cov = pop_cov(data)
    std_div = np.diag(np.sqrt(np.diagonal(cov)))
    ew_cov = pop_ew_cov(data, lamda)
    d1 = np.diag(np.sqrt(np.diagonal(ew_cov)) ** (-1))
    ew_cor = d1 @ ew_cov @ d1
    ew_cor_var = std_div @ ew_cor @ std_div
    return ew_cor_var


def missing_cov(x: pd.DataFrame, skipMiss=True, fun='cov'):
    """
    calculate the covariance or correlation matrix with missing values
    """
    if skipMiss:
        new_x = x.dropna()
        if fun == 'cov':
            return new_x.cov()
        elif fun == 'cor':
            return new_x.corr()
    else:  # pairwise
        if fun == 'cov':
            return x.cov()
        elif fun == 'cor':
            return x.corr()
