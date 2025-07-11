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

def create_calendar_heatmap(df):
    df_copy = df.copy()
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['day'] = df_copy['date'].dt.day
    df_copy['weekday'] = df_copy['date'].dt.dayofweek
    months = sorted(df_copy['month'].unique())
    fig, axes = plt.subplots(len(months), 1, figsize=(14, 5*len(months)))
    if len(months) == 1:
        axes = [axes]
    for i, month in enumerate(months):
        month_data = df_copy[df_copy['month'] == month].copy()
        first_day = datetime(month_data['year'].iloc[0], month, 1)
        first_weekday = first_day.weekday()
        month_data['week_of_month'] = ((month_data['day'] - 1 + first_weekday) // 7)
        pivot_table = month_data.pivot_table(
            values='probability', 
            index='week_of_month', 
            columns='weekday', 
            fill_value=np.nan
        )
        day_pivot_table = month_data.pivot_table(
            values='day', 
            index='week_of_month', 
            columns='weekday', 
            fill_value=np.nan
        )
        annot_array = np.full(pivot_table.shape, '', dtype=object)
        for row in range(pivot_table.shape[0]):
            for col in range(pivot_table.shape[1]):
                if not np.isnan(pivot_table.iloc[row, col]):
                    day_val = int(day_pivot_table.iloc[row, col])
                    prob_val = pivot_table.iloc[row, col]
                    annot_array[row, col] = f'{day_val}\n{prob_val:.3f}'
        sns.heatmap(
            pivot_table, 
            annot=annot_array, 
            fmt='', 
            cmap='Reds', 
            ax=axes[i],
            cbar_kws={'label': '生理確率'},
            xticklabels=['月', '火', '水', '木', '金', '土', '日'],
            mask=pivot_table.isna(),
            annot_kws={'fontsize': 10},
            vmin=0,
            vmax=1
        )
        axes[i].set_title(f'{month_data["year"].iloc[0]}年{month}月の生理確率ヒートマップ')
        axes[i].set_xlabel('曜日')
        axes[i].set_ylabel('週')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_calendar_heatmap_plotly(df):
    # 年月日情報を分解
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday  # 0=月曜
    df['week'] = df['date'].dt.isocalendar().week

    # 月ごとに分割してヒートマップを作成
    months = sorted(df['month'].unique())
    figs_html = ""
    for month in months:
        month_data = df[df['month'] == month]
        # ピボットテーブル（週×曜日）
        pivot = month_data.pivot(index='week', columns='weekday', values='probability')
        days = month_data.pivot(index='week', columns='weekday', values='day')
        # 上下反転（週の昇順→降順）
        pivot = pivot.iloc[::-1]
        days = days.iloc[::-1]
        # アノテーション（日付＋確率）
        annotations = []
        for i, week in enumerate(pivot.index):
            for j, wd in enumerate(pivot.columns):
                val = pivot.iloc[i, j]
                day = days.iloc[i, j]
                if not pd.isna(val):
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=f"{int(day)}<br>{val:.2f}",
                            showarrow=False,
                            font=dict(size=12, color='black')
                        )
                    )
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['月','火','水','木','金','土','日'],
            colorscale='Reds',
            zmin=0, zmax=1,
            colorbar=dict(title='生理確率')
        ))
        fig.update_layout(
            title=f"{month_data['year'].iloc[0]}年{month}月",
            annotations=annotations,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        figs_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    return figs_html

@app.route("/", methods=["GET", "POST"])
def index():
    stats_info = None
    heatmap_html = None
    default_data = "23,24,25,32,25"
    default_start_date = "2025-07-03"
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
            # フォームから開始日を取得（なければデフォルト）
            start_date_str = input_start_date
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            except Exception:
                start_date = datetime.strptime(default_start_date, "%Y-%m-%d")
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
            heatmap_html = create_calendar_heatmap_plotly(df)
        except Exception as e:
            stats_info = {"error": str(e)}
    return render_template("index.html", stats=stats_info, heatmap_html=heatmap_html, default_data=input_data, default_start_date=input_start_date)