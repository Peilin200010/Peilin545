from riskill import option
import datetime


def test_gbsm():
    result = option.gbsm(True, 165, 160, 0.038, 0.0425, 0.0425, 0.2)
    expected = 5.98
    assert abs(result - expected) <= 0.01


underlying = 151.03
strike = 165
rf = 0.0425
divRate = 0.0053
current = datetime.datetime(2022, 3, 13)
expire = datetime.datetime(2022, 4, 15)
calendarDays = (datetime.datetime(2022, 12, 31) - datetime.datetime(2021, 12, 31)).days
ttm = (expire - current).days / calendarDays
b = rf - divRate
ivol = 0.2
N = (expire - current).days


def test_bt_american():
    result = option.bt_american(True, underlying, strike, ttm, rf, b, ivol, N)
    expected = 0.3351
    assert abs(result - expected) <= 0.01


divA = [0.88]
div1 = datetime.datetime(2022, 4, 11)
divT = [(div1 - current).days]


def test_bt_american_with_div():
    result = option.bt_american_with_div(False, underlying, strike, ttm, rf, divA, divT, ivol, N)
    expected = 14.5463
    assert abs(result - expected) <= 0.01


def test_greeks():
    g = option.Greeks(True, underlying, strike, ttm, rf, b, ivol)
    result = g.delta_finite_dif(1)
    expected = 0.0834
    assert abs(result - expected) <= 0.001
    result = g.gamma()
    expected = 0.0168
    assert abs(result - expected) <= 0.001


current = datetime.datetime(2023, 3, 3)
expire = datetime.datetime(2023, 4, 21)
calendarDays = (datetime.datetime(2023, 12, 31) - datetime.datetime(2022, 12, 31)).days
ttm2 = (expire - current).days / calendarDays


def test_find_iv():
    result = option.find_iv(True, underlying, 125, ttm2, rf, b, 27.3)
    expected = 0.3746
    assert abs(result - expected) <= 0.001
