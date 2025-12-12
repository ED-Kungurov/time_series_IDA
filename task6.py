import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

#лаговые признаки
df['lag_1'] = df['value'].shift(1)
df['lag_2'] = df['value'].shift(2)
df['lag_7'] = df['value'].shift(7)
df['lag_14'] = df['value'].shift(14)
df['lag_30'] = df['value'].shift(30)

#скользящие статистики
windows = [7, 14, 30]
for w in windows:
    df[f'rolling_mean_{w}'] = df['value'].shift(1).rolling(window=w).mean()
    df[f'rolling_std_{w}'] = df['value'].shift(1).rolling(window=w).std()
    df[f'rolling_min_{w}'] = df['value'].shift(1).rolling(window=w).min()
    df[f'rolling_max_{w}'] = df['value'].shift(1).rolling(window=w).max()

#временные признаки
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)



# EMA
df['ema_7'] = df['value'].ewm(span=7, adjust=False).mean().shift(1)
df['ema_30'] = df['value'].ewm(span=30, adjust=False).mean().shift(1)

# 2. Удаление пропусков
print(f"Shape до dropna: {df.shape}")
df.dropna(inplace=True)
print(f"Shape после dropna: {df.shape}")

# 3. Разделение на X и y
y = df['value']
X = df.drop(['value'], axis=1)

# 4. Train/test split (временной порядок!)
# shuffle=False критически важно
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# 5. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Превратим обратно в DataFrame для удобства
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

print("Данные подготовлены и масштабированы.")
print(X_train_scaled.head())
