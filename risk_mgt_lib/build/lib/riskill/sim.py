import numpy as np
import pandas as pd
import scipy.stats as sps
import math


class Direct:
    @staticmethod
    def chol_psd(mat: np.ndarray) -> np.ndarray:
        number = mat.shape[0]
        root = np.zeros((number, number))
        for j in range(number):
            s = 0.
            if j > 0:
                s = root[j, :j]@root[j, :j].T
            temp = mat[j, j] - s
            if -1e-8 <= temp <= 0:
                temp = 0.
            root[j, j] = math.sqrt(temp)

            if root[j, j] == 0:
                root[j:, j] = 0.
            else:
                for i in range(j+1, number):
                    s = root[i, :j]@root[j, :j].T
                    root[i, j] = (mat[i, j] - s) / root[j, j]
        return root

    def simulation(self, mat: np.ndarray, draw: int = 25000) -> np.ndarray:
        L = self.chol_psd(mat)
        Z = np.random.standard_normal((L.shape[0], draw))
        X = (L @ Z).T
        return X


class PCA:
    @staticmethod
    def conduct_pca(mat: np.ndarray) -> (np.ndarray, np.ndarray, list):
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        e_vals_sort = np.flip(np.real(eigenvalues), axis=0)
        pos_e_vals = e_vals_sort[e_vals_sort >= 1e-8]
        pos_e_vecs = np.flip(np.real(eigenvectors), axis=1)[:, : pos_e_vals.size]
        cum_explain = [pos_e_vals[:i+1].sum() / pos_e_vals.sum() for i in range(pos_e_vals.size)]
        return pos_e_vals, pos_e_vecs, cum_explain

    def simulation(self, mat: np.ndarray, draw: int = 25000, min_explain: float = 1.0):
        pca = self.conduct_pca(mat)
        pca_vals = pca[0]
        pca_vecs = pca[1]
        for i, cum in enumerate(pca[2]):
            if cum >= min_explain:
                pca_vals = pca_vals[: i + 1]
                pca_vecs = pca_vecs[:, : i + 1]
        B = pca_vecs @ np.diag(np.sqrt(pca_vals))
        r = np.random.standard_normal((B.shape[1], draw))
        X = (B @ r).T
        return X


class Copula:
    @staticmethod
    def simulation_multi_t(mat: pd.DataFrame, draw: int = 25000, min_explain: float = 1.0) -> np.ndarray:
        U = mat.copy()
        frees = []
        mus = []
        sigmas = []

        # transform with CDF to conduct simulation
        for col in mat.columns:
            free, mu, sigma = sps.t.fit(mat[col])
            U[col] = mat[col].apply(lambda x: sps.t.cdf(x, free, mu, sigma))
            frees.append(free)
            mus.append(mu)
            sigmas.append(sigma)

        cor = sps.spearmanr(U)[0]
        N = PCA().simulation(cor, draw, min_explain)

        # transform back
        U_sim = sps.norm.cdf(N)
        X = U_sim.copy()
        for i in range(U_sim.shape[1]):
            X[:, i] = sps.t.ppf(U_sim[:, i], frees[i], loc=mus[i], scale=sigmas[i])

        return X
