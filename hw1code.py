import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

# ------ problem 1 ------

# standard normal distribution, skew = 0, kurt = 0
mu, sigma = 0, 1
def f_std_normal(x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


samples = 1000
s_skew = np.arange(float(samples))
s_kurt = np.arange(float(samples))
for i in range(samples):
    s = np.random.normal(mu, sigma, 10)
    s_skew[i] = st.skew(s)
    s_kurt[i] = st.kurtosis(s)
count, bins, ignored = plt.hist(s_skew, 100, density=True, color='k')
plt.plot(bins, f_std_normal(bins), linewidth=3, color='r')
plt.title("skewness")
plt.show()
count, bins, ignored = plt.hist(s_kurt, 100, density=True, color='k')
plt.plot(bins, f_std_normal(bins), linewidth=3, color='r')
plt.title("kurtosis")
plt.show()
t_value, p_value = st.ttest_1samp(s_skew, 0)
t_value2, p_value2 = st.ttest_1samp(s_kurt, 0)
print("t value, p value for skewness: ", t_value, p_value)
print("t value, p value for kurtosis: ", t_value2, p_value2)


# increase sample size to 100000
samples = 100_000
s_skew = np.arange(float(samples))
s_kurt = np.arange(float(samples))
for i in range(samples):
    s = np.random.normal(mu, sigma, 10)
    s_skew[i] = st.skew(s)
    s_kurt[i] = st.kurtosis(s)
count, bins, ignored = plt.hist(s_skew, 100, density=True, color='k')
plt.plot(bins, f_std_normal(bins), linewidth=3, color='r')
plt.title("skewness")
plt.show()
count, bins, ignored = plt.hist(s_kurt, 100, density=True, color='k')
plt.plot(bins, f_std_normal(bins), linewidth=3, color='r')
plt.title("kurtosis")
plt.show()
t_value, p_value = st.ttest_1samp(s_skew, 0)
t_value2, p_value2 = st.ttest_1samp(s_kurt, 0)
print("t value, p value for skewness: ", t_value, p_value)
print("t value, p value for kurtosis: ", t_value2, p_value2)

# ------ problem 2 ------

df = pd.read_csv('w2project/problem2.csv')
# prepare vector x and y from csv file
X = np.mat(df['x']).T
Y = np.mat(df['y']).T
# calculate beta and error
B = (X.T @ X).I @ X.T @ Y
e = Y - X @ B
print("beta using OLS: ", float(B))
count, bins, ignored = plt.hist(e, 50, density=True, color='k')
plt.plot(bins, f_std_normal(bins), linewidth=3, color='r')
plt.show()
print(st.skew(e), st.kurtosis(e))

from scipy import optimize
import math


def negative_normal_ll(arguments):
    s, beta = arguments[0], arguments[1]
    n = Y.size
    er = Y - X * beta
    s2 = s * s
    ll = -n / 2 * math.log(s2 * 2 * np.pi) - er.T @ er / (2 * s2)
    return -ll


bnds = ((0.0, None), (None, None))
result_normal = optimize.minimize(negative_normal_ll, (1.0, 0.6), bounds=bnds)
normal_beta = result_normal.x[1]
normal_s = result_normal.x[0]
normal_ll = -negative_normal_ll([normal_s, normal_beta])
AIC1 = 2 * 3 - 2 * normal_ll  # k = 3 considering s, beta and intercept
print("Normal beta: ", normal_beta)
print("Normal S: ", normal_s)
print("Normal LL: ", float(normal_ll))
print("Normal AIC: ", float(AIC1))

def __t_loglikelihood(mu, s, free, x):
    n = Y.size
    free12 = (free + 1.0) / 2.0
    part1 = math.lgamma(free12) - math.lgamma(free / 2.0) - math.log(s * math.sqrt(free * np.pi))
    part2 = 1.0 + (1 / free) * (np.multiply((x - mu) / s, (x - mu) / s))
    part2_sum = (np.log(part2)).sum()
    ll = n * part1 - free12 * part2_sum
    return ll


def negative_t_ll(arguments):
    s, free, beta = arguments[0], arguments[1], arguments[2]
    er = Y - X * beta
    ll = __t_loglikelihood(0.0, s, free, er)
    return -ll


bnds2 = ((0.0, None), (3, None), (None, None))
result_t = optimize.minimize(negative_t_ll, (1.0, 10, 0.6), bounds=bnds2)
t_beta = result_t.x[2]
t_free = result_t.x[1]
t_s = result_t.x[0]
t_ll = -negative_t_ll([t_s, t_free, t_beta])
AIC2 = 2 * 4 - 2 * t_ll  # k = 4 considering s, degree of freedom, beta and intercept
print("T distribution beta: ", t_beta)
print("T distribution degree of freedom: ", t_free)
print("T distribution S: ", t_s)
print("T distribution LL: ", float(t_ll))
print("T distribution AIC: ", float(AIC2))

# ------ problem 3 ------
from statsmodels.tsa.stattools import acf, pacf

n = 1000
x = np.arange(n)
err = np.random.randn(n)


def get_AR_n(number):
    y_time_series = np.zeros(n)
    for i in range(number):
        y_time_series[i] = i + 1.0
    for i in range(number, n):
        y_time_series[i] = 1.0 + err[i]
        for j in range(number):
            y_time_series[i] += 0.3 * y_time_series[i-j-1]
    return y_time_series

# AR(1)
AR1 = get_AR_n(1)
plt.plot(x, AR1)
plt.title("AR1")
plt.show()
AR1_ACF = acf(AR1)
plt.plot(AR1_ACF, color='k')
plt.title("ACF of AR1")
plt.show()
AR1_PACF = pacf(AR1)
plt.plot(AR1_PACF, color='k')
plt.title("PACF of AR1")
plt.show()

# AR(2)
AR2 = get_AR_n(2)
plt.plot(x, AR2)
plt.title("AR2")
plt.show()
AR2_ACF = acf(AR2)
plt.plot(AR2_ACF, color='k')
plt.title("ACF of AR2")
plt.show()
AR2_PACF = pacf(AR2)
plt.plot(AR2_PACF, color='k')
plt.title("PACF of AR2")
plt.show()

# AR(3)
AR3 = get_AR_n(3)
plt.plot(x, AR3)
plt.title("AR3")
plt.show()
AR3_ACF = acf(AR3)
plt.plot(AR3_ACF, color='k')
plt.title("ACF of AR3")
plt.show()
AR3_PACF = pacf(AR3)
plt.plot(AR3_PACF, color='k')
plt.title("PACF of AR3")
plt.show()


def get_MA_n(number):
    y_time_series = np.zeros(n)
    for i in range(number):
        y_time_series[i] = 1.0
    for i in range(number, n):
        y_time_series[i] = 1.0 + err[i]
        for j in range(number):
            y_time_series[i] += 5 * err[i-j-1]
    return y_time_series

# MA(1)
MA1 = get_MA_n(1)
plt.plot(x, MA1)
plt.title("MA1")
plt.show()
MA1_ACF = acf(MA1)
plt.plot(MA1_ACF, color='k')
plt.title("ACF of MA1")
plt.show()
MA1_PACF = pacf(MA1)
plt.plot(MA1_PACF, color='k')
plt.title("PACF of MA1")
plt.show()

# MA(2)
MA2 = get_MA_n(2)
plt.plot(x, MA2)
plt.title("MA2")
plt.show()
MA2_ACF = acf(MA2)
plt.plot(MA2_ACF, color='k')
plt.title("ACF of MA2")
plt.show()
MA2_PACF = pacf(MA2)
plt.plot(MA2_PACF, color='k')
plt.title("PACF of MA2")
plt.show()

# MA(3)
MA3 = get_MA_n(3)
plt.plot(x, MA3)
plt.title("MA3")
plt.show()
MA3_ACF = acf(MA3)
plt.plot(MA3_ACF, color='k')
plt.title("ACF of MA3")
plt.show()
MA3_PACF = pacf(MA3)
plt.plot(MA3_PACF, color='k')
plt.title("PACF of MA3")
plt.show()
