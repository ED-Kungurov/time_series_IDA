import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# наастройка для графиков
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

print(df.head())
print(df.describe())

# 1. Постройте график временного ряда
# 2. Добавьте скользящее среднее (rolling mean) с окном 30 дней
# 3. Добавьте доверительный интервал (rolling mean ± 2*rolling std)

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['value'], label='Исходный ряд', alpha=0.7)

#среднее и отклонение c окном в 30 дней используется для того чтобы было лучше видно общую тенденцию без шума
rol_mn = df['value'].rolling(window=30).mean()
rol_std = df['value'].rolling(window=30).std()

plt.plot(rol_mn.index, rol_mn, label='Rolling Mean (30 days)', color='red', linewidth=2)

# Доверительный интервал
plt.fill_between(rol_mn.index, rol_mn - 2 * rol_std, rol_mn + 2 * rol_std, color='gray', alpha=0.2, label='Confidence Interval')

plt.title('Анализ временного ряда: Тренд и Сезонность')
plt.legend()
plt.show()

# 4. Проанализируйте визуально: есть ли тренд? сезонность? выбросы?
print("Визуальный анализ: Наблюдается явный восходящий линейный тренд и сезонные колебания.")

#гистограмма чатсот значений
plt.figure(figsize=(10, 6))
sns.histplot(df['value'], kde=True, bins=30)
plt.title('Распределение значений временного ряда')
plt.show()

#boxplot по месяцам
df['month'] = df.index.month
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='month', y='value')
plt.title('Сезонность')
plt.show()
