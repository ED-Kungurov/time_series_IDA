import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller, kpss

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

np.random.seed(42)
n = 730
dates = pd.date_range(start='2022-01-01', periods=n, freq='D')
time = np.arange(n)
trend = 0.03 * time + 100
seasonal_yearly = 15 * np.sin(2 * np.pi * time / 365)
seasonal_weekly = 5 * np.sin(2 * np.pi * time / 7)
noise = np.random.normal(0, 3, n)
ts_values = trend + seasonal_yearly + seasonal_weekly + noise
df = pd.DataFrame({
    'date': dates,
    'value': ts_values
})
df.set_index('date', inplace=True)

#ADF тест определение стационарности ряда
#стационарный ряд тот у которого среднее, отклонение не меняются со временем
#adf - вероятность случайности считается исходя из предыдущих значений если она меньше 0.05 то она стационарна.
result_adf = adfuller(df['value'].dropna())
print(f'ADF: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')
print('critical values:')
for key, value in result_adf[4].items():
    print(f'\t{key}: {value}')
if result_adf[1] < 0.05:
    print("ряд стационарен")
else:
    print("ряд не стационарен")

#KPSS тест определение стационарности ряда
#kpss если вероятность случайности меньше 0.05 то она не стационарна. Я не понимаю что это значит.
result_kpss = kpss(df['value'].dropna(), regression='c', nlags="auto")
print(f'KPSS Statistic: {result_kpss[0]}')
print(f'p-value: {result_kpss[1]}')
if result_kpss[1] >= 0.05:
    print("стационарен по KPSS")
else:
    print("не стационарен по KPSS")


#дифференцирование - способ сделать его стационарным берем не значения а разницу между ними
df['diff_1'] = df['value'].diff()
plt.figure(figsize=(12, 4))
plt.plot(df['diff_1'])
plt.title("ряд после дифференцирования")
plt.show()
#проверка на стационарность как в ADF
result_adf = adfuller(df['diff_1'].dropna())
print(f'p-value: {result_adf[1]}')
if result_adf[1] < 0.05:
    print("ряд стационарен")
else:
    print("ряд не стационарен")

#логарифмирование - способ сделать его стационарным берем не значения а логарифм от каждого значения
df['log'] = np.log(df['value'])
result_adf = adfuller(df['log'].dropna())
print(f'pvalue: {result_adf[1]}')
if result_adf[1] < 0.05:
    print("ряд стационарен")
else:
    print("ряд не стационарен")

#логарифмирование + дифференцирование
df['log_diff'] = df['log'].diff()
result_adf = adfuller(df['log_diff'].dropna())
print(f'pvalue: {result_adf[1]}')
if result_adf[1] < 0.05:
    print("ряд стационарен")
else:
    print("ряд не стационарен")

