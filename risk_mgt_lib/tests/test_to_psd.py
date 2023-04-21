import numpy as np

from riskill import to_psd


def test_weighted_f_norm():
    data = np.array([[-4, -3, -2],
                     [-1,  0,  1],
                     [ 2,  3,  4]])
    w = np.ones(data.shape[0])
    assert np.isclose(to_psd.weighted_f_norm(data, w), 60)


def test_near_psd():
    data = np.ones((5, 5)) * 0.9
    for i in range(data.shape[0]):
        data[i, i] = 1.0
    data[0, 1] = data[1, 0] = 0.7357
    expected = [[1.000000, 0.735701, 0.899997, 0.899997, 0.899997],
                [0.735701, 1.000000, 0.899997, 0.899997, 0.899997],
                [0.899997, 0.899997, 1.000000, 0.900000, 0.900000],
                [0.899997, 0.899997, 0.900000, 1.000000, 0.900000],
                [0.899997, 0.899997, 0.900000, 0.900000, 1.000000]]
    assert (-1e-5 <= to_psd.near_psd(data) - expected).all() and \
           (to_psd.near_psd(data) - expected <= 1e-5).all()


def test_higham():
    data = np.ones((5, 5)) * 0.9
    for i in range(data.shape[0]):
        data[i, i] = 1.0
    data[0, 1] = data[1, 0] = 0.7357
    w = np.ones(data.shape[0])
    expected = [[1.000000, 0.735704, 0.899998, 0.899998, 0.899998],
                [0.735704, 1.000000, 0.899998, 0.899998, 0.899998],
                [0.899998, 0.899998, 1.000000, 0.900001, 0.900001],
                [0.899998, 0.899998, 0.900001, 1.000000, 0.900001],
                [0.899998, 0.899998, 0.900001, 0.900001, 1.000000]]
    assert (-1e-5 <= to_psd.higham(data, w) - expected).all() and \
           (to_psd.higham(data, w) - expected <= 1e-5).all()
