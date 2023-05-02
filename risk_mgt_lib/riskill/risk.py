import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
import math

from riskill.covmatrix import pop_weights, pop_ew_cov
from riskill.sim import PCA, Copula


def return_calculate(prices: pd.DataFrame, method: str = 'DISCRETE', date_column: str = 'Date') \
        -> pd.DataFrame:
    Vars = prices.columns.drop(date_column)
    n_vars = prices.shape[1]
    if n_vars == Vars.size:
        raise ValueError("date column: " + date_column + " not in DataFrame")
    n_vars -= 1

    p = prices[Vars]
    out = (p / p.shift(1)).iloc[1:, :]

    if method.upper() == "DISCRETE":
        out = out.apply(lambda x: x-1)
    elif method.upper() == "LOG":
        out = out.apply(np.log)
    else:
        raise ValueError("method: " + method + " must be 'LOG' or 'DISCRETE'")

    out.index = prices[date_column][1:]
    return out


def VaR_ES(data, alpha: float = 0.05) -> (float, float):
    data_sort = np.sort(data)
    n = alpha * data.size
    VaR = (data_sort[math.ceil(n)-1] + data_sort[math.floor(n)-1]) / 2
    ES = data_sort[:math.floor(n)-1].mean()
    return -VaR, -ES


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

    @staticmethod
    def VaR_ES_norm(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        mu, sigma = sps.norm.fit(returns)
        sim_t = sps.norm.rvs(loc=mu, scale=sigma, size=10000)
        return VaR_ES(sim_t, alpha)

    @staticmethod
    def VaR_ES_norm_ew(returns: pd.Series or pd.DataFrame, alpha: float = 0.05, lamba: float = 0.94)\
            -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        weights = np.zeros(returns.size)
        weights = pop_weights(weights, lamba)
        mu = returns.mean()
        ew_var = (weights * (returns - mu)).T @ (returns - mu)
        sim_t = sps.norm.rvs(loc=mu, scale=math.sqrt(ew_var), size=10000)
        return VaR_ES(sim_t, alpha)

    @staticmethod
    def VaR_ES_t(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        free, mu, sigma = sps.t.fit(returns)
        sim_t = sps.t.rvs(free, loc=mu, scale=sigma, size=10000)
        return VaR_ES(sim_t, alpha)

    def portfolio_VaR_ES_t(prices: pd.DataFrame, P: dict, alpha: float = 0.05,
                           return_method: str = 'DISCRETE', date_column: str = 'Date') -> (float, float):
        """
        in dollar
        """
        prices_P = prices[[date_column] + list(P.keys())]
        holdings = np.array(list(P.values()))
        current_value_P = holdings.T @ prices_P.iloc[-1, 1:]
        returns_P = (return_calculate(prices_P, return_method)).dropna()
        sim_returns = Copula().simulation_multi_t(returns_P)['X']

        if return_method.upper() == 'DISCRETE':
            sim_prices = (sim_returns + 1) * np.array(prices_P.iloc[-1, 1:])
        elif return_method.upper() == 'LOG':
            sim_prices = np.exp(sim_returns) * np.array(prices_P.iloc[-1, 1:])
        else:
            raise ValueError("method: " + return_method + " must be 'LOG' or 'DISCRETE'")

        sim_value_P = (sim_prices * holdings).sum(axis=1)
        return VaR_ES(sim_value_P - current_value_P, alpha)

    @staticmethod
    def delta_normal_VaR_ES(A: pd.Series or pd.DataFrame, Q: pd.Series or pd.DataFrame,
                            delta: np.ndarray, P: float, var: float, Nday: int=1, alpha: float = 0.05) -> (float, float):
        """
        in dollar, consider only one underlying price, assuming μ=0
        """
        PV = A@Q
        dR = P/PV * delta@Q
        sigma = np.sqrt(dR*var*dR)*np.sqrt(Nday)
        P_ret = sps.norm.rvs(scale=sigma,size=10_000)
        delta_VaR, delta_ES = VaR_ES(P_ret, alpha)
        return PV * delta_VaR, PV * delta_ES


class BySim:
    @staticmethod
    def VaR_ES_AR1(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        returns.index = pd.DatetimeIndex(returns.index).to_period('D')
        result = sm.tsa.AutoReg(returns, lags=1).fit()
        ret_t_1 = returns[-1]
        m = result.params[0]
        beta = result.params[1]
        er_scale = math.sqrt(result.sigma2)

        sim_times = 1_000_000
        sim_ret_t = beta * ret_t_1 + np.random.normal(0, er_scale, sim_times) + m
        return VaR_ES(sim_ret_t, alpha)

    @staticmethod
    def VaR_ES_his(returns: pd.Series or pd.DataFrame, alpha: float = 0.05) -> (float, float):
        if type(returns) == pd.DataFrame and returns.shape[1] > 1:
            raise ValueError("process returns for one asset at a time")
        sim_return = np.random.choice(returns, 10000)
        return VaR_ES(sim_return, alpha)

    @staticmethod
    def portfolio_VaR_ES(prices: pd.DataFrame, P: dict, lamda: float = 0.94, alpha: float = 0.05,
                         return_method='DISCRETE', model='M', date_column: str = 'Date') -> (float, float):
        """
        in dollar
        """
        prices_P = prices[[date_column] + list(P.keys())]
        holdings = np.array(list(P.values()))
        current_value_P = holdings.T @ prices_P.iloc[-1, 1:]
        returns_P = return_calculate(prices_P, return_method)

        # simulate with exponentially weighted covariance
        if model.upper() == 'M':
            ew_cov = pop_ew_cov(returns_P, lamda)
            sim_returns = PCA().simulation(ew_cov) + np.array(returns_P.mean())
        elif model.upper() == 'H':
            w = np.zeros(returns_P.shape[0])
            row_draw = np.random.choice(returns_P.shape[0], 25000, p=pop_weights(w, lamda))
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
        return VaR_ES(sim_value_P - current_value_P, alpha)
