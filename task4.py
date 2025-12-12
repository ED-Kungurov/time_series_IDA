import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

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

d = 1
df_diff = df['value'].diff().dropna()

#ACF и PACF графики для стационарного ряда
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(df_diff, ax=plt.gca(), lags=40)
plt.title('ACF') #автокорреляция насколько сильно сегодня зависит от вчера

plt.subplot(1, 2, 2)
plot_pacf(df_diff, ax=plt.gca(), lags=40)
plt.title('PACF') #частичная автокорреляция, автокорреляция убирая влияние промежуточных дней
plt.tight_layout()
plt.show()

print("Анализ графиков: p определяется по PACF, q по ACF.")

#для каждого варианта обучение ARIMA модельки
#популярный алгоритм для прогноза врем рядов
#разделение на train/test 80/20
train_size = int(len(df) * 0.8)
train = df['value'][:train_size]
test = df['value'][train_size:]
results = []

#перебор параметров p, q, d они же авторегрессия, интегрирование, скользящее среднее
for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            bic = model_fit.bic
            results.append({'p': p, 'd': d, 'q': q, 'AIC': aic, 'BIC': bic})
            print(f"ARIMA({p},{d},{q}) - AIC:{aic:.2f}")
        except Exception as e:
            print(f"ARIMA({p},{d},{q}) - Error: {e}")

#сравнение моделей по AIC и BIC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='AIC')

print(results_df.head())

best_params = results_df.iloc[0]
print(f"\nлучшая:({int(best_params['p'])}, {int(best_params['d'])}, {int(best_params['q'])})")
