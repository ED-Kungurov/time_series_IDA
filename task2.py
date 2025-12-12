import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose

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

#аддитивная декомозиция с периодом 365
#для показа размаха сезонных колебаний
result_add = seasonal_decompose(df['value'], model='additive', period=365)
result_add.plot()
plt.suptitle('аддитивная декомпозиция', y=1.02)
plt.show()

#мультипликативная модель с периодом 365
#тоже для показа размаха сезонных колебаний, но тут уже перемножение а не сложение
result_mult = seasonal_decompose(df['value'], model='multiplicative', period=365)
result_mult.plot()
plt.suptitle('мультипликативная декомпозиция', y=1.02)
plt.show()

#анализ остатков аддитивной модели
resid_add = result_add.resid.dropna()
resid_mult = result_mult.resid.dropna()


print(f"средний остатоток аддитивной модели: {resid_add.mean():.4f}")
print(f"стандартное отклонение аддитивной модели: {resid_add.std():.4f}")
#тест на нормальность
stat, p = stats.normaltest(resid_add)
print(f"normaltest: {p:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(resid_add, kde=True, bins=30)
plt.title('распределение остатков')
plt.subplot(1, 2, 2)
stats.probplot(resid_add, dist="norm", plot=plt)
plt.title('QQ plot')
plt.show()

print(f"средний остатоток мультипликативной модели: {resid_mult.mean():.4f}")
print(f"стандартное отклонение мультипликативной модели: {resid_mult.std():.4f}")
stat, p = stats.normaltest(resid_mult)
print(f"normaltest: {p:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(resid_mult, kde=True, bins=30)
plt.title('распределение остатков')
plt.subplot(1, 2, 2)
stats.probplot(resid_mult, dist="norm", plot=plt)
plt.title('QQ plot')
plt.show()

#6 аддитивная модель лучше так как остатки более стабильны
