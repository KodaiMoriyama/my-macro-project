import pandas as pd
import numpy as np
import pandas_datareader.data as web
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
jp_font_path = 'C:\Windows\Fonts\msgothic.ttc'
jp_font = fm.FontProperties(fname=jp_font_path)
plt.rcParams['font.family'] = jp_font.get_name()

start_date = '1955-01-01'
end_date = '2022-01-01'

# データ取得
gdp_germany = web.DataReader('NAEXKP01DEQ189S', 'fred', start=start_date, end=end_date)
gdp_japan = web.DataReader('NAEXKP01JPQ189S', 'fred', start=start_date, end=end_date)

# 正しく1列目を抽出してSeriesに変換
data = pd.DataFrame({
    'Germany': gdp_germany.iloc[:, 0],
    'Japan': gdp_japan.iloc[:, 0]
})

# 欠損値の処理
data.dropna(inplace=True)

# 自然対数変換
log_data = np.log(data)

# λ = 1600（四半期データ用）
cycle_germany, trend_germany = hpfilter(log_data['Germany'], lamb=1600)
cycle_japan, trend_japan = hpfilter(log_data['Japan'], lamb=1600)

# 結果の保存
trend_df = pd.DataFrame({
    'Germany_trend': trend_germany,
    'Japan_trend': trend_japan
})

cycle_df = pd.DataFrame({
    'Germany_cycle': cycle_germany,
    'Japan_cycle': cycle_japan
})

# ① 標準偏差の算出
std_germany = cycle_df['Germany_cycle'].std()
std_japan = cycle_df['Japan_cycle'].std()

# ② 相関係数の算出
correlation = cycle_df['Germany_cycle'].corr(cycle_df['Japan_cycle'])

# 結果表示
print(f"ドイツの循環成分の標準偏差: {std_germany:.4f}")
print(f"日本の循環成分の標準偏差: {std_japan:.4f}")
print(f"ドイツと日本の循環成分の相関係数: {correlation:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(cycle_df.index, cycle_df['Germany_cycle'], label='Germany (Cycle)', color='blue')
plt.plot(cycle_df.index, cycle_df['Japan_cycle'], label='Japan (Cycle)', color='red')
plt.title('ドイツと日本のGDP循環成分の比較', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Log-deviation from Trend')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
