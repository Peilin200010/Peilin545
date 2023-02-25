import numpy as np
import pandas as pd
import scipy.stats as sps
import math

from riskill.covmatrix import pop_weights, pop_ew_cov
from riskill.sim import PCA


def return_calculate(prices: pd.DataFrame, method: str = 'DISCRETE', date_column: str = 'Date') \
        -> pd.DataFrame:
    Vars = prices.columns.drop(date_column)
    n_vars = prices.shape[1]
    if n_vars == Vars.size:
        raise ValueError("date column: " + date_column + " not in DataFrame")
    n_vars -= 1

    p = prices[Vars]
    out = (p / p.shift(1)).dropna()

    if method.upper() == "DISCRETE":
        out = out.apply(lambda x: x-1)
    elif method.upper() == "LOG":
        out = out.apply(np.log)
    else:
        raise ValueError("method: " + method + " must be 'LOG' or 'DISCRETE'")

    out.index = prices[date_column][1:]
    return out


class ByDistri:
    @staticmethod
    def VaR_norm(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> float:
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        mu = returns.mean()
        sigma = (np.array(returns).var()) ** (1 / 2)
        VaR = -sps.norm.ppf(alpha, loc=mu, scale=sigma)
        return VaR

    @staticmethod
    def VaR_norm_ew(returns: pd.Series or pd.DataFrame, alpha: float = 0.05, lamba: float = 0.94)\
            -> float:
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        weights = np.zeros(returns.size)
        weights = pop_weights(weights, lamba)
        mu = returns.mean()
        ew_var = (weights * (returns - mu)).T @ (returns - mu)
        VaR = -sps.norm.ppf(alpha, loc=mu, scale=ew_var ** (1 / 2))
        return VaR

    @staticmethod
    def VaR_t(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> float:
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        free, mu, sigma = sps.t.fit(returns)
        VaR = -sps.t.ppf(alpha, free, loc=mu, scale=sigma)
        return VaR


class BySim:
    @staticmethod
    def portfolio_VaR(prices: pd.DataFrame, P: dict, lamda: float = 0.94, alpha: float = 0.05,
                      return_method='DISCRETE', model='M', date_column: str = 'Date') -> float:
        prices_P = prices[[date_column] + list(P.keys())]
        holdings = np.array(list(P.values()))
        current_value_P = holdings.T @ prices_P.iloc[-1, 1:]
        returns_P = return_calculate(prices_P, return_method)

        # simulate with exponentially weighted covariance
        weight = np.zeros(returns_P.shape[0])
        w = pop_weights(weight, lamda)
        if model.upper() == 'M':
            ew_cov = pop_ew_cov(w, returns_P)
            sim_returns = PCA.simulation(ew_cov) + np.array(returns_P.mean())
        elif model.upper() == 'H':
            row_draw = np.random.choice(returns_P.shape[0], 25000, p=w)
            sim_returns = np.array(returns_P.iloc[row_draw, :])
        else:
            raise ValueError("model: " + model + "must be 'M' or 'H'")

        if return_method.upper() == 'DISCRETE':
            sim_prices = (sim_returns + 1) * np.array(prices_P.iloc[-1, 1:])
        elif return_method.upper() == 'LOG':
            sim_prices = np.exp(sim_returns) * np.array(prices_P.iloc[-1, 1:])
        else:
            raise ValueError("method: " + return_method + " must be 'LOG' or 'DISCRETE'")
        sim_value_P = (sim_prices * holdings).sum(axis=1)
        VaR_P = current_value_P - np.percentile(sim_value_P, alpha * 100)
        return VaR_P

    @staticmethod
    def VaR_ES(data, alpha: float = 0.05) -> (float, float):
        data_sort = np.sort(data)
        n = alpha * data.size
        VaR = (data_sort[math.ceil(n)] + data_sort[math.floor(n)]) / 2
        ES = data_sort[:math.floor(n)].mean()
        return -VaR, -ES

    def VaR_ES_t(self, returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        free, mu, sigma = sps.t.fit(returns)
        sim_t = sps.t.rvs(free, loc=mu, scale=sigma, size=10000)
        return self.VaR_ES(sim_t, alpha)
