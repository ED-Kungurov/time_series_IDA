import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

train_size = int(len(df) * 0.8)
train = df['value'][:train_size]
test = df['value'][train_size:]
#лучшие значения по предыдущему заданию
p, d, q = 2, 1, 3
print(f"Обучаем модель ARIMA({p},{d},{q})...")

model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

#summary
print(model_fit.summary())

#диагностика остатков
residuals = model_fit.resid
print(f"остатки среднее: {residuals.mean():.4f}")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(residuals)
plt.title('остатки')

plt.subplot(2, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('остатки плотность')

plt.subplot(2, 2, 3)
plot_acf(residuals, ax=plt.gca())
plt.title('ACF остатков')

plt.subplot(2, 2, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ')

plt.tight_layout()
plt.show()

#прогноз на тестовый период
forecast_res = model_fit.get_forecast(steps=len(test))
forecast_values = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

#визуализация
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='green')
plt.plot(forecast_values.index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_values.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.title(f'ARIMA({p},{d},{q}) прогноз vs реальность')
plt.legend()
plt.show()

#метрики: mae, rmse, mape
mae = mean_absolute_error(test, forecast_values)
rmse = np.sqrt(mean_squared_error(test, forecast_values))
mape = np.mean(np.abs((test - forecast_values) / test)) * 100

print("Метрики:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
