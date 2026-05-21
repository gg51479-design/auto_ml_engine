import warnings
import numpy as np
from utils.logger import log

def fit_forecast(model_name, y_log, steps, seasonal_periods):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "SARIMAX":
            try:
                import pmdarima as pm
                model = pm.auto_arima(
                    y_log, seasonal=True, m=seasonal_periods,
                    stepwise=True, suppress_warnings=True, error_action="ignore"
                )
                return np.asarray(model.predict(n_periods=steps), dtype=float)
            except ImportError:
                log.warning("ไม่ได้ติดตั้ง pmdarima ระบบใช้ SARIMAX แบบดั้งเดิม")
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(y_log, order=(1, 1, 1), seasonal_order=(1, 1, 0, seasonal_periods))
                return np.asarray(model.fit(disp=False).forecast(steps=steps), dtype=float)

        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(y_log, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
        return np.asarray(model.fit(optimized=True).forecast(steps=steps), dtype=float)

def seasonal_naive_forecast(y_real, steps, seasonal_periods):
    seed = y_real[-seasonal_periods:] if len(y_real) >= seasonal_periods else y_real
    return np.asarray(np.resize(seed, steps), dtype=float)

def moving_average_forecast(y_real, steps, seasonal_periods):
    window = min(len(y_real), max(seasonal_periods * 3, 5))
    return np.full(steps, float(np.mean(y_real[-window:])), dtype=float)

def seasonal_median_forecast(y_real, steps, seasonal_periods):
    if len(y_real) < seasonal_periods: return moving_average_forecast(y_real, steps, seasonal_periods)
    history = np.asarray(y_real, dtype=float)
    result = []
    for step in range(steps):
        values = history[-min(len(history), seasonal_periods * 8):][step % seasonal_periods::seasonal_periods]
        result.append(float(np.median(values if len(values) > 0 else history[-seasonal_periods:])))
    return np.asarray(result, dtype=float)

def weighted_recent_pattern_forecast(y_real, steps, seasonal_periods):
    if len(y_real) < seasonal_periods: return moving_average_forecast(y_real, steps, seasonal_periods)
    recent_mean = float(np.mean(y_real[-min(len(y_real), max(seasonal_periods * 4, 6)):]))
    last_pattern = np.resize(y_real[-seasonal_periods:], steps)
    median_pattern = seasonal_median_forecast(y_real, steps, seasonal_periods)
    return (last_pattern * 0.50) + (median_pattern * 0.30) + (recent_mean * 0.20)

def forecast_basic_model(model_name, y_real, steps, seasonal_periods):
    if model_name == "Moving Average": return moving_average_forecast(y_real, steps, seasonal_periods)
    if model_name == "Seasonal Median": return seasonal_median_forecast(y_real, steps, seasonal_periods)
    if model_name == "Weighted Recent Pattern": return weighted_recent_pattern_forecast(y_real, steps, seasonal_periods)
    if model_name == "Seasonal Naive": return seasonal_naive_forecast(y_real, steps, seasonal_periods)