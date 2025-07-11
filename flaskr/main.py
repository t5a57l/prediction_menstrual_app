import matplotlib
matplotlib.use('Agg')
# 日本語用にmatplotlibのjapanizeフォントを設定
import japanize_matplotlib
from flask import render_template, request, redirect, url_for, send_file
from flaskr import app
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import io
import base64
import plotly.graph_objects as go

def normal_pdf(x, mean, std):
    return stats.norm.pdf(x, mean, std)

def calculate_menstrual_probability_cyclic(days_since_last, mean_cycle, std_cycle):
    total_prob = 0
    for cycle_num in range(1, 4):
        expected_start = cycle_num * mean_cycle
        prob = normal_pdf(days_since_last, expected_start, std_cycle)
        total_prob += prob
    return total_prob

def calculate_menstrual_period_probability(days_since_last, mean_cycle, std_cycle, period_duration=7):
    if days_since_last < period_duration:
        return 1.0
    total_prob = 0
    for i in range(period_duration):
        start_day = days_since_last - i
        if start_day > 0:
            start_prob = calculate_menstrual_probability_cyclic(start_day, mean_cycle, std_cycle)
            total_prob += start_prob
    return total_prob

def create_calendar_json(df):
    # 確率値(0-1)を赤系グラデーション色に変換する関数
    def prob_to_color(prob):
        # 0: #fff0f0 (薄いピンク) → 1: #e74c3c (赤)
        # 線形補間
        def lerp(a, b, t):
            return int(a + (b - a) * t)
        # RGB値
        r1, g1, b1 = 255, 240, 240  # #fff0f0
        r2, g2, b2 = 231, 76, 60    # #e74c3c
        r = lerp(r1, r2, prob)
        g = lerp(g1, g2, prob)
        b = lerp(b1, b2, prob)
        return f"#{r:02x}{g:02x}{b:02x}"

    events = []
    for _, row in df.iterrows():
        prob = row['probability']
        color = prob_to_color(prob)
        events.append({
            'title': f"{prob:.2f}",
            'start': row['date'].strftime('%Y-%m-%d'),
            'allDay': True,
            'backgroundColor': color,
            'borderColor': color
        })
    return events

@app.route("/", methods=["GET", "POST"])
def index():
    stats_info = None
    default_data = "24,24,25,28,25"
    default_start_date = "2025-07-01"
    input_data = default_data
    input_start_date = default_start_date
    if request.method == "POST":
        try:
            input_data = request.form.get("cycle_data", default_data)
            input_start_date = request.form.get("start_date", default_start_date)
            cycle_data = np.array([int(x) for x in input_data.split(",")])
            mean_cycle = np.mean(cycle_data)
            std_cycle = np.std(cycle_data, ddof=1)
            variance_cycle = np.var(cycle_data, ddof=1)
            stats_info = {
                "mean": f"{mean_cycle:.2f}",
                "std": f"{std_cycle:.2f}",
                "var": f"{variance_cycle:.2f}"
            }
        except Exception as e:
            stats_info = {"error": str(e)}
    return render_template("index.html", stats=stats_info, default_data=input_data, default_start_date=input_start_date)
# APIエンドポイント: カレンダー用データをJSONで返す
@app.route("/api/calendar_data", methods=["POST"])
def api_calendar_data():
    try:
        input_data = request.json.get("cycle_data", "23,24,25,32,25")
        input_start_date = request.json.get("start_date", "2025-07-03")
        cycle_data = np.array([int(x) for x in input_data.split(",")])
        mean_cycle = np.mean(cycle_data)
        std_cycle = np.std(cycle_data, ddof=1)
        variance_cycle = np.var(cycle_data, ddof=1)
        start_date = datetime.strptime(input_start_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=90)
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        probabilities = []
        for date in dates:
            days_since_last = (date - start_date).days
            if days_since_last == 0:
                prob = 1.0
            else:
                prob = calculate_menstrual_period_probability(days_since_last, mean_cycle, std_cycle)
            probabilities.append(prob)
        df = pd.DataFrame({
            'date': dates,
            'probability': probabilities,
            'days_since_last': [(date - start_date).days for date in dates]
        })
        events = create_calendar_json(df)
        stats = {
            "mean": f"{mean_cycle:.2f}",
            "std": f"{std_cycle:.2f}",
            "var": f"{variance_cycle:.2f}"
        }
        return {"events": events, "stats": stats}
    except Exception as e:
        return {"error": str(e)}, 400