import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

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
df['lag_1'] = df['value'].shift(1)
df['lag_2'] = df['value'].shift(2)
df['lag_7'] = df['value'].shift(7)
df['lag_14'] = df['value'].shift(14)
df['lag_30'] = df['value'].shift(30)
windows = [7, 14, 30]
for w in windows:
    df[f'rolling_mean_{w}'] = df['value'].shift(1).rolling(window=w).mean()
    df[f'rolling_std_{w}'] = df['value'].shift(1).rolling(window=w).std()
    df[f'rolling_min_{w}'] = df['value'].shift(1).rolling(window=w).min()
    df[f'rolling_max_{w}'] = df['value'].shift(1).rolling(window=w).max()
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
df['ema_7'] = df['value'].ewm(span=7, adjust=False).mean().shift(1)
df['ema_30'] = df['value'].ewm(span=30, adjust=False).mean().shift(1)
df.dropna(inplace=True)
y = df['value']
X = df.drop(['value'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

results_summary = {}
models = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}
results = {}
predictions = {}

tscv = TimeSeriesSplit(n_splits=5)
#обучения трех разных моделей ridge random forest gradient boosting
for name, model in models.items():
    #CV - рандомные данные в тренировку и валидацию
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_absolute_error')
    #тренировка
    model.fit(X_train_scaled, y_train)
    #прогноз
    pred = model.predict(X_test_scaled)
    predictions[name] = pred
    # метрики
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    r2 = r2_score(y_test, pred)
    results[name] = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

#обучение arima
arima_model = ARIMA(y_train, order=(2, 1, 3))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(y_test))
arima_pred.index = y_test.index
#стата arima
mae_arima = mean_absolute_error(y_test, arima_pred)
rmse_arima = np.sqrt(mean_squared_error(y_test, arima_pred))
mape_arima = np.mean(np.abs((y_test - arima_pred) / y_test)) * 100
r2_arima = r2_score(y_test, arima_pred)
results_summary['ARIMA'] = {'MAE': mae_arima, 'RMSE': rmse_arima, 'MAPE': mape_arima, 'R2': r2_arima}

#таблица сравнения
results_df = pd.DataFrame(results_summary).T
print("итоговая таблица:")
print(results_df)

#визуализация метрик
results_df[['MAE', 'RMSE', 'MAPE']].plot(kind='bar', figsize=(10, 6))
plt.title('сравнение метрик ошибок моделей')
plt.ylabel('error value')
plt.xticks(rotation=45)
plt.show()

#график остатков у лучшей модели
best_model_name = results_df['MAE'].idxmin()
print(f"лучшая модель: {best_model_name}")

if best_model_name == 'ARIMA':
    best_residuals = arima_fit.resid
    best_test_residuals = y_test - arima_pred
else:
    model = models[best_model_name]
    pred = model.predict(X_test_scaled)
    best_test_residuals = y_test - pred

plt.figure(figsize=(12, 4))
plt.plot(y_test.index, best_test_residuals)
plt.title(f'остатки на тестовой ({best_model_name})')
plt.axhline(0, color='r', linestyle='--')
plt.show()

plt.figure(figsize=(6, 4))
plot_acf(best_test_residuals, ax=plt.gca(), lags=20)
plt.title(f'acf остатков ({best_model_name})')
plt.show()

print("лучшая модель:", best_model_name)
print("ml модели random forest, gradient boosting обычно показывают себя лучше на сложных данных без нелинейностей")
print("arima класс для простых одномерных рядов но требует со стационарностью")
