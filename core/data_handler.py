from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.logger import log

MAX_INPUT_ROWS = 10_000
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DRAW_DATES_TEMPLATE = [
    (1, 17), (2, 1), (2, 16), (3, 1), (3, 16), (4, 1), (4, 16),
    (5, 2), (5, 16), (6, 1), (6, 16), (7, 1), (7, 16), (8, 1), (8, 16),
    (9, 1), (9, 16), (10, 1), (10, 16), (11, 1), (11, 16), (12, 1), (12, 16), (12, 30),
]
MONTH_ALIASES = {
    "1": 1, "01": 1, "jan": 1, "january": 1, "มค": 1, "ม.ค": 1, "มกราคม": 1,
    "2": 2, "02": 2, "feb": 2, "february": 2, "กพ": 2, "ก.พ": 2, "กุมภาพันธ์": 2,
    "3": 3, "03": 3, "mar": 3, "march": 3, "มีค": 3, "มี.ค": 3, "มีนาคม": 3,
    "4": 4, "04": 4, "apr": 4, "april": 4, "เมย": 4, "เม.ย": 4, "เมษายน": 4,
    "5": 5, "05": 5, "may": 5, "พค": 5, "พ.ค": 5, "พฤษภาคม": 5,
    "6": 6, "06": 6, "jun": 6, "june": 6, "มิย": 6, "มิ.ย": 6, "มิถุนายน": 6,
    "7": 7, "07": 7, "jul": 7, "july": 7, "กค": 7, "ก.ค": 7, "กรกฎาคม": 7,
    "8": 8, "08": 8, "aug": 8, "august": 8, "สค": 8, "ส.ค": 8, "สิงหาคม": 8,
    "9": 9, "09": 9, "sep": 9, "sept": 9, "september": 9, "กย": 9, "ก.ย": 9, "กันยายน": 9,
    "10": 10, "oct": 10, "october": 10, "ตค": 10, "ต.ค": 10, "ตุลาคม": 10,
    "11": 11, "nov": 11, "november": 11, "พย": 11, "พ.ย": 11, "พฤศจิกายน": 11,
    "12": 12, "dec": 12, "december": 12, "ธค": 12, "ธ.ค": 12, "ธันวาคม": 12,
}

def smart_transform(series):
    """แปลงข้อมูลโดยรองรับกรณีที่มีค่าติดลบ"""
    min_val = np.min(series)
    shift_value = 0
    if min_val < 0:
        shift_value = abs(min_val) + 1
        transformed = np.log1p(series + shift_value)
    else:
        transformed = np.log1p(series)
    return transformed, shift_value

def smart_inverse_transform(transformed_series, shift_value):
    """แปลงข้อมูลกลับเป็นสเกลเดิม"""
    inversed = np.expm1(transformed_series)
    if shift_value > 0:
        inversed = inversed - shift_value
    return inversed

def normalize_month(value):
    if pd.isna(value): raise ValueError("เดือนว่าง")
    if isinstance(value, (int, np.integer)) or (isinstance(value, (float, np.floating)) and float(value).is_integer()):
        month_num = int(value)
    else:
        key = str(value).strip().lower().replace(".", "").replace(" ", "")
        month_num = MONTH_ALIASES.get(key)
    if month_num is None or month_num < 1 or month_num > 12:
        raise ValueError(f"เดือนไม่ถูกต้อง: {value}")
    return month_num

def parse_row_date(day, month, year):
    year_num = int(year)
    if 2400 <= year_num <= 2600: year_num -= 543
    return datetime(year=year_num, month=normalize_month(month), day=int(day))

def read_input_file(file_path):
    df = pd.read_csv(file_path, header=None) if file_path.lower().endswith(".csv") else pd.read_excel(file_path, header=None)
    if df.shape[1] < 4: raise ValueError("ไฟล์ต้องมีอย่างน้อย 4 คอลัมน์: Value, Day, Month, Year")
    if len(df) > MAX_INPUT_ROWS:
        log.warning(f"ตัดทิ้งส่วนเกิน ไฟล์มี {len(df)} แถว")
        raise ValueError(f"ระบบรองรับสูงสุด {MAX_INPUT_ROWS:,} แถว")
    df = df.iloc[:, :4].copy()
    df.columns = ["Value", "Day", "Month", "Year"]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value", "Day", "Month", "Year"]).reset_index(drop=True)
    if len(df) < 10: raise ValueError("ข้อมูลต้องมีอย่างน้อย 10 แถว")

    dates = []
    for index, row in df.iterrows():
        dates.append(parse_row_date(row["Day"], row["Month"], row["Year"]))
    df["Date"] = dates
    return df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

def build_regular_daily_series(df):
    series = df.set_index("Date")["Value"].sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index, fill_value=0.0).astype(float), series.index.max()

def build_draw_calendar(start_year, end_year):
    dates = []
    for year in range(start_year, end_year + 1):
        for month, day in DRAW_DATES_TEMPLATE:
            try: dates.append(datetime(year=year, month=month, day=day))
            except ValueError: continue
    return sorted(dates)

def is_lottery_draw_date(date_value):
    date_obj = date_value.to_pydatetime() if hasattr(date_value, "to_pydatetime") else date_value
    return (date_obj.month, date_obj.day) in DRAW_DATES_TEMPLATE

def build_irregular_lottery_series(df):
    valid_draws = df[df["Date"].apply(is_lottery_draw_date)].copy()
    if len(valid_draws) < 10: raise ValueError("โหมดปฏิทินงวดไทยต้องมีข้อมูลอย่างน้อย 10 แถว")
    series = valid_draws.sort_values("Date").set_index("Date")["Value"].astype(float)
    return series, series.index.max()

def make_future_labels(last_date, steps, is_irregular):
    labels, current = [], last_date.to_pydatetime() if hasattr(last_date, "to_pydatetime") else last_date
    if is_irregular:
        calendar = build_draw_calendar(current.year, current.year + 5)
        future_dates = [d for d in calendar if d > current][:steps]
        labels = [f"งวดวันที่ {f.day:02d} {MONTH_ORDER[f.month - 1]} {f.year}" for f in future_dates]
    else:
        labels = [f"{(current + timedelta(days=s)).strftime('%Y-%m-%d')} ({DAY_ORDER[(current + timedelta(days=s)).weekday()]})" for s in range(1, steps + 1)]
    return labels