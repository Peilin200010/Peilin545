import numpy as np
import scipy.stats as sps
from scipy.optimize import fsolve


def gbsm(call: bool, underlying, strike, ttm, rf, b, ivol) -> float:
    d1 = (np.log(underlying/strike) + (b + ivol**2 / 2) * ttm) / (ivol * np.sqrt(ttm))
    d2 = d1 - ivol * np.sqrt(ttm)
    if call:
        return underlying*np.exp((b-rf)*ttm)*sps.norm.cdf(d1) - strike*np.exp(-rf*ttm)*sps.norm.cdf(d2)
    elif not call:
        return strike*np.exp(-rf*ttm)*sps.norm.cdf(-d2) - underlying*np.exp((b-rf)*ttm)*sps.norm.cdf(-d1)
    else:
        raise ValueError("call must be True or False")


def bt_american(call: bool, underlying, strike, ttm, rf, b, ivol, N: int) -> float:
    """
    N should the number of steps
    """
    dt = ttm/N
    u = np.exp(ivol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b*dt)-d)/(u-d)
    pdown = 1. - pu
    df = np.exp(-rf*dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return int((n+1)*(n+2)/2)

    def idxFunc(i, j):
        return int(nNodeFunc(j-1)+i)

    nNodes = nNodeFunc(N)
    optionValues = np.zeros(nNodes)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u**i * d**(j-i)
            optionValues[idx] = max(0, z*(price-strike))

            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1, j+1)]
                                        + pdown*optionValues[idxFunc(i, j+1)]))

    return optionValues[0]


def bt_american_with_div(call: bool, underlying, strike, ttm, rf, divAmts: list, divTimes: list,
                         ivol, N: int) -> float:
    if not (len(divAmts) and len(divTimes)):
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)
    elif divTimes[0] > N:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, N)

    dt = ttm/N
    u = np.exp(ivol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(rf*dt)-d)/(u-d)
    pdown = 1. - pu
    df = np.exp(-rf*dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return int((n+1)*(n+2)/2)

    def idxFunc(i, j):
        return int(nNodeFunc(j-1)+i)

    nNodes = nNodeFunc(divTimes[0])
    optionValues = np.zeros(nNodes)

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u**i * d**(j-i)

            if j < divTimes[0]:
                optionValues[idx] = max(0, z*(price-strike))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1, j+1)]
                                        + pdown*optionValues[idxFunc(i, j+1)]))
            else:
                # time of dividend (recursively)
                valNoExercise = bt_american_with_div(call, price-divAmts[0], strike, ttm-divTimes[0]*dt,
                                                     rf, divAmts[1:], [x-divTimes[0] for x in divTimes[1:]],
                                                     ivol, N-divTimes[0])
                valExercise = max(0, z*(price-strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]


def find_iv(call:bool, underlying, strike, ttm, rf, b, option_price, N=None,
            divAmts:list=[], divTimes:list=[], guess=0.2):
    if N is not None and len(divAmts) and len(divTimes):
        def equation(iv):
            return bt_american_with_div(call, underlying, strike, ttm, rf, divAmts, divTimes, iv, N) - option_price
    elif N is not None and not (len(divAmts) and len(divTimes)):
        def equation(iv):
            return bt_american(call, underlying, strike, ttm, rf, b, iv, N) - option_price
    else:
        def equation(iv):
            return gbsm(call, underlying, strike, ttm, rf, b, iv) - option_price
    sol = fsolve(equation, guess)
    return float(sol)


class Greeks:
    def __init__(self, call:bool, underlying, strike, ttm, rf, b, ivol,  N=None,
                 divAmts:list=[], divTimes:list=[]):
        self._call = call
        self._underlying = underlying
        self._strike = strike
        self._ttm = ttm
        self._rf = rf
        self._b = b
        self._ivol = ivol
        self._N = N
        self._divAmts = divAmts
        self._divTimes = divTimes

    def d1(self):
        return (np.log(self._underlying/self._strike) + (self._b + self._ivol**2 / 2) * self._ttm) / (self._ivol * np.sqrt(self._ttm))

    def d2(self):
        return self.d1() - self._ivol * np.sqrt(self._ttm)

    def delta(self):
        """
        first derivative of value wrt underlying
        """
        if self._call:
            return np.exp((self._b - self._rf)*self._ttm) * sps.norm.cdf(self.d1())
        else:
            return np.exp((self._b - self._rf)*self._ttm) * (sps.norm.cdf(self.d1())-1)

    def delta_finite_dif(self, step:float):
        """
        first finite difference of value wrt underlying
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            p1 = bt_american_with_div(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
            p2 = bt_american_with_div(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._b, self._ivol, self._N)
            p2 = bt_american(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._b, self._ivol, self._N)
        else:
            p1 = gbsm(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._b, self._ivol)
            p2 = gbsm(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._b, self._ivol)

        return (p1-p2) / (2*step)

    def gamma(self):
        """
        second derivative of value wrt underlying (convexity)
        """
        nominator = np.exp((self._b - self._rf)*self._ttm) * sps.norm.pdf(self.d1())
        denominator = self._underlying*self._ivol*np.sqrt(self._ttm)
        return nominator / denominator

    def gamma_finite_dif(self, step:float):
        """
        second finite difference of value wrt underlying (convexity)
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            p1 = bt_american_with_div(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
            p2 = bt_american_with_div(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
            p3 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._b, self._ivol, self._N)
            p2 = bt_american(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._b, self._ivol, self._N)
            p3 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol, self._N)
        else:
            p1 = gbsm(self._call, self._underlying+step, self._strike, self._ttm, self._rf, self._b, self._ivol)
            p2 = gbsm(self._call, self._underlying-step, self._strike, self._ttm, self._rf, self._b, self._ivol)
            p3 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol)

        return (p1+p2-2*p3) / (step**2)

    def vega(self):
        """
        first derivative of value wrt implied volatility
        """
        return self._underlying*np.exp((self._b - self._rf)*self._ttm)*sps.norm.pdf(self.d1())*np.sqrt(self._ttm)

    def vega_finite_dif(self, step:float):
        """
        first finite difference of value wrt implied volatility
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            p1 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol+step, self._N)
            p2 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm, self._rf, self._divAmts, self._divTimes, self._ivol-step, self._N)
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol+step, self._N)
            p2 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol-step, self._N)
        else:
            p1 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol+step)
            p2 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b, self._ivol-step)
        return (p1-p2) / (2*step)

    def theta(self):
        """
        first derivative of value wrt ttm
        """
        p1 = -self._underlying*np.exp((self._b - self._rf)*self._ttm)*sps.norm.pdf(self.d1())*self._ivol / (2*np.sqrt(self._ttm))
        p2 = (self._b - self._rf)*self._underlying*np.exp((self._b - self._rf)*self._ttm)
        p3 = self._rf*self._strike*np.exp(-self._rf*self._ttm)
        if self._call:
            return p1-p2*sps.norm.cdf(self.d1())-p3*sps.norm.cdf(self.d2())
        else:
            return p1+p2*sps.norm.cdf(-self.d1())+p3*sps.norm.cdf(-self.d2())

    def theta_finite_dif(self, step:float):
        """
        first finite difference of value wrt ttm
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            p1 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm+step, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
            p2 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm-step, self._rf, self._divAmts, self._divTimes, self._ivol, self._N)
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying, self._strike, self._ttm+step, self._rf, self._b, self._ivol, self._N)
            p2 = bt_american(self._call, self._underlying, self._strike, self._ttm-step, self._rf, self._b, self._ivol, self._N)
        else:
            p1 = gbsm(self._call, self._underlying, self._strike, self._ttm+step, self._rf, self._b, self._ivol)
            p2 = gbsm(self._call, self._underlying, self._strike, self._ttm-step, self._rf, self._b, self._ivol)

        return -(p1-p2) / (2*step)

    def rho(self):
        """
        first derivative of value wrt rf where rf=b
        """
        p1 = self._ttm*self._strike*np.exp(-self._rf*self._ttm)
        if self._call:
            return p1*sps.norm.cdf(self.d2())
        else:
            return -p1*sps.norm.cdf(-self.d2())

    def rho_finite_dif(self, step:float):
        """
        first finite difference of value wrt rf
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            p1 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm, self._rf+step, self._divAmts, self._divTimes, self._ivol, self._N)
            p2 = bt_american_with_div(self._call, self._underlying, self._strike, self._ttm, self._rf-step, self._divAmts, self._divTimes, self._ivol, self._N)
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf+step, self._b, self._ivol, self._N)
            p2 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf-step, self._b, self._ivol, self._N)
        else:
            p1 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf+step, self._b, self._ivol)
            p2 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf-step, self._b, self._ivol)

        return (p1-p2) / (2*step)

    def carry_rho(self):
        """
        first derivative of value wrt b
        """
        p1 = self._ttm*self._underlying*np.exp((self._b - self._rf)*self._ttm)
        if self._call:
            return p1*sps.norm.cdf(self.d1())
        else:
            return -p1*sps.norm.cdf(-self.d1())

    def carry_rho_finite_dif(self, step:float):
        """
        first finite difference of value wrt b
        """
        if self._N is not None and len(self._divAmts) and len(self._divTimes):
            raise ValueError("option with discrete dividends does not have carry rho")
        elif self._N is not None and not (len(self._divAmts) and len(self._divTimes)):
            p1 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b+step, self._ivol, self._N)
            p2 = bt_american(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b-step, self._ivol, self._N)
        else:
            p1 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b+step, self._ivol)
            p2 = gbsm(self._call, self._underlying, self._strike, self._ttm, self._rf, self._b-step, self._ivol)

        return (p1-p2) / (2*step)