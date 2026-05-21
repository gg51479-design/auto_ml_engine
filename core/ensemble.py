import numpy as np
from core.models import forecast_basic_model, fit_forecast, seasonal_naive_forecast
from core.data_handler import smart_transform, smart_inverse_transform
from utils.logger import log

ACCURACY_WARN_THRESHOLD = 20.0
ACCURACY_FALLBACK_THRESHOLD = 5.0

def forecast_with_selected_model(model_name, y_real, steps, seasonal_periods, allow_ensemble=True):
    if model_name in {"Moving Average", "Seasonal Median", "Weighted Recent Pattern", "Seasonal Naive"}:
        return forecast_basic_model(model_name, y_real, steps, seasonal_periods)
    if model_name == "Ensemble":
        return ensemble_forecast(y_real, steps, seasonal_periods) if allow_ensemble else seasonal_naive_forecast(y_real, steps, seasonal_periods)
    
    y_log, shift_val = smart_transform(y_real)
    forecast_log = fit_forecast(model_name, y_log, steps, seasonal_periods)
    return smart_inverse_transform(forecast_log, shift_val)

def ensemble_forecast(y_real, steps, seasonal_periods):
    names = ["Moving Average", "Seasonal Median", "Weighted Recent Pattern", "Seasonal Naive"]
    forecasts, weights = [], []
    for name in names:
        try:
            mae, _, _ = validation_score(name, y_real, seasonal_periods, allow_ensemble=False)
            forecasts.append(forecast_with_selected_model(name, y_real, steps, seasonal_periods))
            weights.append(1.0 / max(mae, 1e-6))
        except: continue
    
    if not forecasts: return seasonal_naive_forecast(y_real, steps, seasonal_periods)
    weights_array = np.asarray(weights, dtype=float) / np.sum(weights)
    return np.average(np.vstack(forecasts), axis=0, weights=weights_array)

def validation_score(model_name, y_real, seasonal_periods, allow_ensemble=True):
    n = len(y_real)
    test_size = min(5, max(1, n // 5))
    if n - test_size < max(seasonal_periods * 2, 10): test_size = max(1, min(3, n // 5))
    train, test = y_real[:-test_size], y_real[-test_size:]
    if len(train) < 4 or len(test) == 0: raise ValueError("ข้อมูลไม่พอสำหรับ validation")

    forecast = forecast_with_selected_model(model_name, train, len(test), seasonal_periods, allow_ensemble)
    mae = float(np.mean(np.abs(test - forecast)))
    
    mase_denom = float(np.mean(np.abs(train[seasonal_periods:] - train[:-seasonal_periods]))) if len(train) > seasonal_periods else 0.0
    if mase_denom < 1e-6: mase_denom = float(np.std(train)) or 1.0
    
    mase = mae / mase_denom
    accuracy = max(0.0, min(100.0, (1 - mase) * 100))
    return mae, accuracy, test_size

def choose_model(selected_model, y_real, seasonal_periods):
    model_names = [selected_model] if selected_model != "ให้ระบบเลือกโมเดลที่ดีที่สุด (Auto)" else ["SARIMAX", "Holt-Winters", "Moving Average", "Seasonal Median", "Weighted Recent Pattern", "Ensemble"]
    candidates = []
    
    for name in model_names:
        try:
            candidates.append((*validation_score(name, y_real, seasonal_periods), name))
        except Exception as e: log.debug(f"{name} failed: {e}")
            
    if not candidates: return "Seasonal Naive", 0.0, "โมเดลล้มเหลว", "fallback"
    
    mae, accuracy, test_size, model_name = min(candidates, key=lambda item: item[0])
    base_info = f"validation MAE = {mae:.3f} ({accuracy:.2f}%)\nเลือกจาก holdout {test_size} งวด"
    
    if accuracy < ACCURACY_FALLBACK_THRESHOLD:
        try:
            fb_mae, fb_acc, fb_test = validation_score("Weighted Recent Pattern", y_real, seasonal_periods)
            return "Weighted Recent Pattern", fb_acc, f"⚠️ Accuracy ต่ำ ระบบ Fallback สำเร็จ", "fallback"
        except: pass
        
    warning_level = "warn" if accuracy < ACCURACY_WARN_THRESHOLD else "ok"
    return model_name, accuracy, base_info, warning_level