import numpy as np
import sys


def weighted_f_norm(A: np.ndarray, W: np.ndarray) -> float:
    # set w = I if unweighted
    WAW = np.diag(np.sqrt(W))@A@np.diag(np.sqrt(W))
    return np.linalg.norm(WAW)**2


def near_psd(mat: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    number = mat.shape[0]
    invSD = None
    result = mat.copy()

    # calculate the correlation matrix if we got a covariance
    if np.diagonal(result).sum() != number:
        invSD = np.diag(np.sqrt(np.diagonal(result))**(-1))
        result = invSD@result@invSD

    vals, vecs = np.linalg.eigh(result)
    vals = np.maximum(vals, np.ones(vals.shape)*epsilon)
    T = (vecs*vecs@vals)**(-1)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T@vecs@l
    result = B@B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(np.diagonal(invSD)**(-1))
        result = invSD@result@invSD

    return result


def first_projection(A: np.ndarray) -> np.ndarray:
    PuA = A.copy()
    for i in range(PuA.shape[0]):
        PuA[i, i] = 1.
    return PuA


def second_projection(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    WAW = np.diag(np.sqrt(W))@A@np.diag(np.sqrt(W))
    vals, vecs = np.linalg.eigh(WAW)
    vals = np.maximum(vals, np.zeros(vals.shape))
    PsA = np.diag(np.sqrt(W)**(-1))@(vecs@np.diag(vals)@vecs.T)@np.diag(np.sqrt(W)**(-1))
    return PsA


def higham(A: np.ndarray, W: np.ndarray, max_iteration: int = 100)-> np.ndarray:
    dif_S = np.zeros(A.shape)
    last_gama = sys.float_info.max

    # calculate the correlation matrix if we got a covariance
    invSD = None
    if np.diagonal(A).sum() != A.shape[0]:
        invSD = np.diag(np.sqrt(np.diagonal(A))**(-1))
        A = invSD@A@invSD

    Y = A.copy()

    for i in range(max_iteration):
        R = Y - dif_S
        X = second_projection(R, W)
        dif_S = X - R
        Y = first_projection(X)
        cur_gama = weighted_f_norm(Y - A, W)
        evals = np.linalg.eigh(Y)[0]
        if abs(cur_gama - last_gama) <= 1e-9 and evals[0] >= -1e-8:
            break
        last_gama = cur_gama

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(np.diagonal(invSD)**(-1))
        Y = invSD@Y@invSD

    return Y
