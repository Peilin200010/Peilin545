import numpy as np
import pandas as pd

from riskill import covmatrix as cvm


def test_pop_weights():
    w = np.zeros(3)
    expected = np.array([1/7.0, 2/7.0, 4/7.0])
    assert (cvm.pop_weights(w, 0.5) == expected).all()


def test_pop_ew_cov():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [6, 5, 4]})
    expected = np.array([[0.714285714, -0.714285714], [-0.714285714, 0.714285714]])
    assert (-1e-5 <= cvm.pop_ew_cov(df, 0.5) - expected).all() and \
           (cvm.pop_ew_cov(df, 0.5) - expected <= 1e-5).all()


def test_pop_cov():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [6, 5, 4]})
    expected = np.array([[2/3.0, -2/3.0], [-2/3.0, 2/3.0]])
    assert (cvm.pop_cov(df) == expected).all()


def test_pop_cor_ew_var():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [6, 5, 4]})
    expected = np.array([[0.714285714, -0.714285714], [-0.714285714, 0.714285714]])
    assert (-1e-5 <= cvm.pop_cor_ew_var(df, 0.5) - expected).all() and \
           (cvm.pop_cor_ew_var(df, 0.5) - expected <= 1e-5).all()


def test_pop_ew_cor_var():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [6, 5, 4]})
    expected = np.array([[2 / 3.0, -2 / 3.0], [-2 / 3.0, 2 / 3.0]])
    assert (cvm.pop_ew_cor_var(df, 0.5) == expected).all()
