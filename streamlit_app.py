import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import threading

# นำเข้าฟังก์ชันจาก Core Engine ของคุณ
from core.data_handler import (
    read_input_file, build_irregular_lottery_series, 
    build_regular_daily_series, make_future_labels, MONTH_ORDER
)
from core.ensemble import choose_model, forecast_with_selected_model
from utils.logger import log

APP_TITLE = "Analytics Auto-ML Universal Hybrid Engine V17.0"
AUTO_MODEL_LABEL = "ให้ระบบเลือกโมเดลที่ดีที่สุด (Auto)"

# ─── ตั้งค่าหน้าเพจ ───────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# กำหนด Session State เพื่อเก็บข้อมูลระหว่างรัน
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None

# ─── ส่วนหัวและคำอธิบาย ───────────────────────────────────────────
st.title("Predictive Auto-ML Cyber Engine 2026")
st.markdown("---")

# ─── Sidebar: การตั้งค่า (UI) ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    
    # 1. โหมดข้อมูล
    is_irregular = st.radio(
        "เลือกชนิดข้อมูล",
        ("ข้อมูลทั่วไปแบบรายวัน", "ข้อมูลปฏิทินงวดไทย")
    ) == "ข้อมูลปฏิทินงวดไทย"
    
    # 2. อัปโหลดไฟล์
    uploaded_file = st.file_uploader("นำเข้าไฟล์ฐานข้อมูล (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # เพื่อให้ใช้งานกับฟังก์ชันเดิมที่รับ path เราจะอ่านเป็น DataFrame โดยตรงก่อน
            # (Note: ฟังก์ชัน read_input_file เดิมรับ path, ใน Streamlit เราอ่านเป็น DataFrame ก่อนได้ หรือจะปรับให้รับ DataFrame โดยตรงก็ได้)
             if uploaded_file.name.endswith(".csv"):
                 df_temp = pd.read_csv(uploaded_file, header=None)
             else:
                 df_temp = pd.read_excel(uploaded_file, header=None)
                 
             # แปลง DataFrame ให้เข้าฟอร์แมตที่ระบบต้องการ
             df_temp = df_temp.iloc[:, :4].copy()
             df_temp.columns = ["Value", "Day", "Month", "Year"]
             df_temp["Value"] = pd.to_numeric(df_temp["Value"], errors="coerce")
             df_temp = df_temp.dropna(subset=["Value", "Day", "Month", "Year"]).reset_index(drop=True)
             
             # ใช้ฟังก์ชันช่วยในการแปลงวันที่ (สมมติว่าปรับให้อ่านจาก DataFrame ได้แล้ว)
             from core.data_handler import parse_row_date
             dates = []
             for index, row in df_temp.iterrows():
                 dates.append(parse_row_date(row["Day"], row["Month"], row["Year"]))
             df_temp["Date"] = dates
             st.session_state.df = df_temp.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
             
             st.success(f"โหลดข้อมูลสำเร็จ ({len(st.session_state.df)} แถว)")
        except Exception as e:
            st.error(f"ข้อผิดพลาดในการโหลดไฟล์: {e}")

    # 3. เลือกโมเดล
    model_options = [AUTO_MODEL_LABEL, "SARIMAX", "Holt-Winters", "Moving Average", "Seasonal Median", "Weighted Recent Pattern", "Ensemble"]
    selected_model = st.selectbox("โมเดลคำนวณ", model_options)
    
    # 4. จำนวนงวด
    steps = st.number_input("จำนวนลำดับที่ต้องการพยากรณ์ (1-30)", min_value=1, max_value=30, value=5)
    
    # ปุ่มรันระบบ
    run_button = st.button("🚀 รันระบบประมวลผล", use_container_width=True, type="primary")

# ─── ส่วนหลัก: ประมวลผลและแสดงผล ─────────────────────────────────────
if run_button:
    if st.session_state.df is None:
        st.warning("กรุณาอัปโหลดไฟล์ฐานข้อมูลก่อนรันระบบ")
    else:
        with st.spinner("กำลังประมวลผล..."):
            try:
                # 1. จัดเตรียมข้อมูลตามโหมดที่เลือก
                df_snap = st.session_state.df.copy()
                if is_irregular:
                    series, last_date = build_irregular_lottery_series(df_snap)
                    sp, mode_info = 2, "โหมดปฏิทินงวดไทย"
                else:
                    series, last_date = build_regular_daily_series(df_snap)
                    sp, mode_info = 7, "โหมดข้อมูลทั่วไปแบบรายวัน"
                
                y_real = series.to_numpy(dtype=float)
                
                # 2. เลือกรันโมเดล (ใช้ฟังก์ชันจาก Core Engine)
                model_name, acc, val_info, warn = choose_model(selected_model, y_real, sp)
                preds = forecast_with_selected_model(model_name, y_real, steps, sp)
                labels = make_future_labels(last_date, steps, is_irregular)
                
                # เก็บผลลัพธ์ลง Session State
                st.session_state.results = {
                    "y_real": y_real,
                    "preds": preds,
                    "labels": labels,
                    "model_name": model_name,
                    "acc": acc,
                    "val_info": val_info,
                    "warn": warn,
                    "mode_info": mode_info,
                    "last_date": last_date,
                    "is_irregular": is_irregular
                }
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
                log.error(f"Streamlit Error: {e}")

# ─── แสดงผลลัพธ์ ──────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results
    
    # 1. กล่องแสดงสถานะความแม่นยำ
    st.subheader(f"📊 ผลการประมวลผล: {res['model_name']}")
    
    if res['warn'] == "fallback":
        st.error(f"**ความแม่นยำจาก holdout:** {res['acc']:.2f}% (ต่ำมาก - ระบบ Fallback ไปยัง Weighted Recent Pattern)\n\n{res['val_info']}")
    elif res['warn'] == "warn":
        st.warning(f"**ความแม่นยำจาก holdout:** {res['acc']:.2f}%\n\n{res['val_info']}")
    else:
        st.success(f"**ความแม่นยำจาก holdout:** {res['acc']:.2f}%\n\n{res['val_info']}")
        
    st.info(res['mode_info'])

    # 2. แสดงผลกราฟ (ใช้ Plotly เพื่อให้ Interactive สวยงามบนเว็บ)
    st.markdown("### 📈 กราฟพยากรณ์อนาคต")
    
    # จัดเตรียมข้อมูลสำหรับกราฟ
    display_n = min(len(res['y_real']), 120)
    hist_y = res['y_real'][-display_n:]
    
    fig = go.Figure()
    
    if res['is_irregular']:
        # แกน X เป็นลำดับ (Sequence)
        hist_x = list(range(display_n))
        future_x = [display_n + i for i in range(len(res['preds']))]
        
        fig.add_trace(go.Scatter(x=hist_x, y=hist_y, mode='lines+markers', name='Historical', line=dict(color='#60A5FA')))
        # จุดเชื่อมระหว่างอดีตกับอนาคต
        fig.add_trace(go.Scatter(x=[hist_x[-1]] + future_x, y=[hist_y[-1]] + list(res['preds']), mode='lines+markers', name='Forecast', line=dict(color='#10B981', dash='dash')))
        
    else:
        # แกน X เป็นวันที่จริง
        current = res['last_date'].to_pydatetime() if hasattr(res['last_date'], "to_pydatetime") else res['last_date']
        hist_dates = [current - timedelta(days=display_n - 1 - i) for i in range(display_n)]
        future_dates = [current + timedelta(days=i) for i in range(1, len(res['preds']) + 1)]
        
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_y, mode='lines', name='Historical', line=dict(color='#60A5FA')))
        fig.add_trace(go.Scatter(x=[hist_dates[-1]] + future_dates, y=[hist_y[-1]] + list(res['preds']), mode='lines+markers', name='Forecast', line=dict(color='#10B981', dash='dash')))

    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 3. แสดงตารางผลลัพธ์ และปุ่ม Export
    st.markdown("### 📋 ตารางพยากรณ์อนาคต")
    
    # สร้าง DataFrame สำหรับผลลัพธ์
    df_results = pd.DataFrame({
        "งวด/วันที่": res['labels'],
        "ค่าที่คาดการณ์": [max(0, int(round(float(p)))) for p in res['preds']]
    })
    
    st.dataframe(df_results, use_container_width=True)
    
    # ปุ่มดาวน์โหลด (Export) 
    csv = df_results.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="💾 ดาวน์โหลดผลลัพธ์ (CSV)",
        data=csv,
        file_name='forecast_results.csv',
        mime='text/csv',
    )