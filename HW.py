import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

start_date = '1955-01-01'
end_date = '2022-01-01'
gdp_raw = web.DataReader('CLVMNACSCAB1GQDE', 'fred', start=start_date, end=end_date)
gdp_clean = gdp_raw.dropna()
gdp_log = np.log(gdp_clean)

lambdas = [10, 100, 1600]
trend_dict = {}
cycle_dict = {}

for lam in lambdas:
    cycle, trend = hpfilter(gdp_log.squeeze(), lamb=lam)
    trend_dict[lam] = trend
    cycle_dict[lam] = cycle

plt.figure(figsize=(12, 6))
plt.plot(gdp_log, label='Log Real GDP', color='black', linewidth=2)

colors = ['blue', 'green', 'red']
for lam, color in zip(lambdas, colors):
    plt.plot(trend_dict[lam], label=f'Trend (λ={lam})', linestyle='--', color=color)

plt.title('Log Real GDP and HP Filter Trends (Germany)')
plt.xlabel('Year')
plt.ylabel('Log GDP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

for lam, color in zip(lambdas, colors):
    plt.plot(cycle_dict[lam], label=f'Cycle (λ={lam})', linestyle='-', color=color)

plt.title('HP Filter Cyclical Components (Germany Log Real GDP)')
plt.xlabel('Year')
plt.ylabel('Deviation from Trend')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()