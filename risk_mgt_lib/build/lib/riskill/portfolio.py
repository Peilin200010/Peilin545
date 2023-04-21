import numpy as np
import pandas as pd
from scipy.optimize import minimize

from riskill.risk import VaR_ES


def optimize_risk(stockMeans, covar, R: float) -> (np.ndarray, float):
    def objective(w): return w.T @ covar @ w
    x0 = [1 / stockMeans.size] * stockMeans.size
    bnds = [(0, None)] * stockMeans.size
    def constraint1(w): return sum(w) - 1
    def constraint2(w): return w @ stockMeans - R
    cons = ({'type': 'eq', 'fun': constraint1}, {'type': 'eq', 'fun': constraint2})
    result = minimize(objective, x0, bounds=bnds, constraints=cons)
    weight = result.x
    risk = np.sqrt(weight.T @ covar @ weight)
    return weight, risk


def optimize_sharpe(stockMeans, covar, rf=0.0025):
    def objective(w): return -(w.T @ stockMeans - rf) / np.sqrt(w.T @ covar @ w)
    x0 = [1 / stockMeans.size] * stockMeans.size
    bnds = [(0, None)] * stockMeans.size
    def constraint(w): return w.sum() - 1.
    cons = ({'type': 'eq', 'fun': constraint})
    result = minimize(objective, x0, bounds=bnds, constraints=cons)
    weight = result.x
    return weight


def weights_through_time(returns: pd.DataFrame, stocks: list, weight) -> (np.ndarray, np.ndarray):
    n = returns.shape[0]
    m = len(stocks)

    pReturn = np.zeros(n)
    weights = np.zeros((n, m))
    lastW = weight.copy()
    matReturns = returns[stocks]

    for i in range(n):
        weights[i, :] = lastW
        lastW = lastW * (1. + matReturns.iloc[i, :])
        pR = lastW.sum()
        lastW = lastW / pR
        pReturn[i] = pR - 1
    return weights, pReturn


def carino(pReturn):
    totalRet = np.exp(np.log(pReturn + 1.).sum()) - 1
    k = np.log(totalRet + 1.) / totalRet
    carinoK = np.log(1. + pReturn) / pReturn / k
    return carinoK


def ex_post_attribution(returns: pd.DataFrame, stocks: list, weight):
    matReturns = returns[stocks]
    weights, pReturn = weights_through_time(returns, stocks, weight)

    returns['Portfolio'] = pReturn
    carinoK = carino(pReturn)
    attrib = (matReturns * weights).mul(carinoK, axis='index')

    Attribution = pd.DataFrame(index=['TotalReturn', 'Return Attribution'])

    # return attribution
    for s in stocks + ['Portfolio']:
        # total stock return
        tr = np.exp(np.log(returns[s] + 1.).sum()) - 1
        if s == 'Portfolio':
            atr = tr
        else:
            atr = attrib[s].sum()
        Attribution[s] = [tr, atr]

    # risk attribution
    Y = matReturns * weights
    X = np.concatenate((np.ones((pReturn.size, 1)), pReturn.reshape(-1, 1)), axis=1)
    B = (np.matrix(X.T @ X).I @ X.T @ Y).iloc[1, :]

    cSD = B * pReturn.std(ddof=1)
    cSD['Portfolio'] = pReturn.std(ddof=1)
    Attribution.loc['Vol Attribution'] = cSD

    return Attribution


def factor_weights_through_time(returns: pd.DataFrame, stocks: list, weight,
                                ffData: pd.DataFrame, Betas: np.ndarray):
    n = returns.shape[0]
    m = len(stocks)

    pReturn = np.zeros(n)
    residReturn = np.zeros(n)
    weights = np.zeros((n, m))
    factorWeights = np.zeros((n, len(ffData.columns)))
    lastW = weight.copy()
    matReturns = returns[stocks]
    ffReturns = ffData.copy()

    for i in range(n):
        weights[i, :] = lastW
        factorWeights[i, :] = np.sum(Betas * np.array(lastW).reshape(-1, 1), axis=0)
        lastW = lastW * (1. + matReturns.iloc[i, :])
        pR = lastW.sum()
        lastW = lastW / pR
        pReturn[i] = pR - 1
        residReturn[i] = (pR - 1.) - factorWeights[i, :] @ ffReturns.iloc[i, :]

    return factorWeights, residReturn, pReturn


def ex_post_factor_attribution(returns: pd.DataFrame, stocks: list, weight,
                               ffData: pd.DataFrame, Betas: np.ndarray):
    ffReturns = ffData.copy()
    factorWeights, residReturn, pReturn = factor_weights_through_time(returns, stocks, weight, ffData, Betas)
    ffData['Alpha'] = residReturn
    ffData['Portfolio'] = pReturn
    carinoK = carino(pReturn)

    attrib = (ffReturns * factorWeights).mul(carinoK, axis='index')
    attrib['Alpha'] = residReturn * carinoK

    # return attribution
    Attribution = pd.DataFrame(index=['TotalReturn', 'Return Attribution'])
    newFactors = list(ffData.columns) + ['Alpha']
    for s in newFactors + ['Portfolio']:
        # total stock return
        tr = np.exp(np.log(ffData[s] + 1.).sum()) - 1
        if s == 'Portfolio':
            atr = tr
        else:
            atr = attrib[s].sum()
        Attribution[s] = [tr, atr]

    # risk attribution
    Y = pd.concat([ffReturns * factorWeights, pd.DataFrame(residReturn,
                                                           index=ffReturns.index,
                                                           columns=['Alpha'])],
                  axis=1)
    X = np.concatenate((np.ones((pReturn.size, 1)), pReturn.reshape(-1, 1)), axis=1)
    B = (np.matrix(X.T @ X).I @ X.T @ Y).iloc[1, :]
    cSD = B * pReturn.std(ddof=1)
    cSD['Portfolio'] = pReturn.std(ddof=1)
    Attribution.loc['Vol Attribution'] = cSD

    return Attribution


def risk_budget(covar, weight):
    """
    ex-ante risk attribution
    """
    pSig = np.sqrt(weight.T @ covar @ weight)
    CSD = (weight * (covar @ weight)) / pSig
    return CSD / pSig


def risk_parity(covar):
    def pvol(w):
        return np.sqrt(w.T @ covar @ w)

    def pCSD(w):
        pVol = pvol(w)
        return (w * (covar @ w)) / pVol

    def sseCSD(w):
        csd = pCSD(w)
        mCSD = csd.sum() / w.size
        dCsd = csd - mCSD
        se = dCsd * dCsd
        return 1e5 * se.sum()  # apply a multiplier

    n = covar.shape[0]
    x0 = [1 / n] * n
    bnds = [(0, None)] * n
    constraint = lambda w: w.sum() - 1.
    cons = ({'type': 'eq', 'fun': constraint})
    result = minimize(sseCSD, x0, bounds=bnds, constraints=cons)

    return result.x


def risk_parity_ES(returns):
    def CES(w):
        n = w.size
        ces = np.zeros(n)
        es = VaR_ES(returns @ w)[1]
        e = 1e-6
        for i in range(n):  # finite difference
            old = w[i]
            w[i] += e
            ces[i] = old * (VaR_ES(returns @ w)[1] - es) / e
            w[i] = old
        return ces

    def SSE_CES(w):
        ces = CES(w)
        ces -= ces.mean()
        return 1e3 * (ces @ ces)

    n = returns.shape[1]
    x0 = [1 / n] * n
    bnds = [(0, None)] * n
    constraint = lambda w: w.sum() - 1.
    cons = ({'type': 'eq', 'fun': constraint})
    result = minimize(SSE_CES, x0, bounds=bnds, constraints=cons)

    return result.x
