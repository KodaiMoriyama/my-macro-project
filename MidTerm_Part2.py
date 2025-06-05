import warnings
warnings.filterwarnings("ignore")        # 将来警告を非表示

import numpy as np
import pandas as pd

# PWT 10.01 データを取得
try:
    import py4macro
    df = py4macro.data('pwt')
except ModuleNotFoundError:
    url = "https://www.rug.nl/ggdc/docs/pwt100.dta"
    df  = pd.read_stata(url)

# OECD22カ国リスト
oecd = [
    "Australia","Austria","Belgium","Canada","Denmark","Finland","France",
    "Germany","Greece","Iceland","Ireland","Italy","Japan","Netherlands",
    "New Zealand","Norway","Portugal","Spain","Sweden","Switzerland",
    "United Kingdom","United States"
]

# 期間フィルタ
period_start, period_end = 1990, 2019
years = period_end - period_start

df = df.loc[
    (df["country"].isin(oecd)) &
    (df["year"].between(period_start, period_end))
].copy()

# 基本変数を計算
df["alpha"] = 1 - df["labsh"]                 # 資本分配率 α
df["y_pc"]  = df["rgdpna"] / df["emp"]        # 一人当たり GDP (Y/N)
df["k_pc"]  = df["rkna"]   / df["emp"]        # 一人当たり資本 (K/N)

# 各国の成長率・寄与度を算出
def growth_rate(series):
    """対数差÷年数 → % 表示の平均成長率"""
    return 100 * np.log(series.iloc[-1] / series.iloc[0]) / years

records = []
for c, g in df.groupby("country"):
    # 連続対数平均成長率
    g_y   = growth_rate(g["y_pc"])
    g_k   = growth_rate(g["k_pc"])
    alpha = g.loc[g["year"].isin([period_start, period_end]), "alpha"].mean()

    # 成長会計
    capital_deepening = alpha * g_k           # 資本深化 (寄与)
    tfp_growth        = g_y - capital_deepening
    tfp_contrib       = tfp_growth / g_y   # g_TFP / g_Y

    records.append({
        "Country": c,
        "Growth Rate": round(g_y,  2),
        "TFP Growth": round(tfp_growth, 2),
        "Capital Deepening": round(capital_deepening, 2),
        "TFP Share": round(tfp_contrib, 2),
        "Capital Share": round(alpha, 3)
    })

# DataFrame 化して並べ替え・平均行を追加
result = (
    pd.DataFrame(records)
      .set_index("Country")
      .sort_values("Growth Rate", ascending=False)
)

# 全体平均行
avg = result.mean().to_frame().T
avg.index = ["Average"]
result = pd.concat([result, avg])  
result = pd.concat([result.drop("Average"), avg])       # 平均行を一時的に除外
result = result.sort_index()                             # 国名でABC順に並べ替え
result = pd.concat([result, avg])                        # 平均行を末尾に再追加

# 表示
print("\nOECD Growth Accounting (1990-2019)")
print("="*85)
print(result.to_string(float_format="%.2f"))
