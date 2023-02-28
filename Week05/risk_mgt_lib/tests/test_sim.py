import numpy as np

from riskill import sim

def test_chol_psd():
    data = np.array([[1., 0.9, 0.9],
                     [0.9, 1., 0.9],
                     [0.9, 0.9, 1.]])
    root = sim.Direct.chol_psd(data)
    assert np.isclose(root@root.T, data).all()

def test_conduct_pca():
    data = np.array([[1., 0.9, 0.9],
                     [0.9, 1., 0.9],
                     [0.9, 0.9, 1.]])
    expected1 = np.array([2.8, 0.1, 0.1])
    expected2 = np.array([[0.57735027, -0.21768015,  0.78694474],
                          [0.57735027,  0.79035421, -0.20495583],
                          [0.57735027, -0.57267406, -0.58198891]])
    expected3 = [2.8/3, 2.9/3, 1.]
    result = sim.PCA.conduct_pca(data)
    assert np.isclose(result[0], expected1).all() and \
           np.isclose(result[1], expected2).all() and \
           np.isclose(result[2], expected3).all()
