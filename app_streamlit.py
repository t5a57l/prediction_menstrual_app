import streamlit as st  # Streamlit本体
import numpy as np      # 数値計算用
import pandas as pd     # データフレーム操作用
from datetime import datetime, timedelta  # 日付操作
from scipy import stats  # 統計計算
import plotly.graph_objects as go  # グラフ描画

# --- ヘルプページの内容を読み込む ---
def load_help():
    try:
        with open("help.md", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ヘルプの読み込みに失敗しました: {e}"

# --- サイドバーでページ切り替え ---
page = st.sidebar.radio("ページ選択", ("カレンダー", "ヘルプ"))

if page == "ヘルプ":
    st.title("ヘルプ")
    st.markdown(load_help())
    st.stop()  # 以降の処理を止める

# --- 計算ロジック ---

def normal_pdf(x, mean, std):
    # 正規分布の確率密度関数を返す
    return stats.norm.pdf(x, mean, std)

def calculate_menstrual_probability_cyclic(days_since_last, mean_cycle, std_cycle):
    # 周期性を考慮して、生理開始日の確率を計算
    total_prob = 0  # 合計確率の初期化
    for cycle_num in range(1, 4):  # 3周期分まで考慮
        expected_start = cycle_num * mean_cycle  # 予想される生理開始日
        prob = normal_pdf(days_since_last, expected_start, std_cycle)  # その日の確率
        total_prob += prob  # 合計に加算
    return total_prob  # 合計確率を返す

def calculate_menstrual_period_probability(days_since_last, mean_cycle, std_cycle, period_duration=7):
    if days_since_last < period_duration:
        return 1.0
    prob_not_period = 1.0
    for i in range(period_duration):
        start_day = days_since_last - i
        if start_day > 0:
            p_start = calculate_menstrual_probability_cyclic(start_day, mean_cycle, std_cycle)
            prob_not_period *= (1 - p_start)
    # 生理期間中でない確率を引いて生理期間中の確率を計算
    prob_period = 1 - prob_not_period
    return prob_period  

def prob_to_color(prob):
    # 確率値(0-1)を赤系グラデーション色に変換
    def lerp(a, b, t):
        return int(a + (b - a) * t)  # 線形補間
    r1, g1, b1 = 255, 240, 240  # 開始色（薄ピンク）
    r2, g2, b2 = 231, 76, 60    # 終了色（赤）
    r = lerp(r1, r2, prob)  # R成分
    g = lerp(g1, g2, prob)  # G成分
    b = lerp(b1, b2, prob)  # B成分
    return f'rgb({r},{g},{b})'  # RGB形式で返す

def create_calendar_heatmap_plotly(df):
    # 月ごとにヒートマップ形式で生理確率を可視化
    df['year'] = df['date'].dt.year  # 年を抽出
    df['month'] = df['date'].dt.month  # 月を抽出
    df['day'] = df['date'].dt.day  # 日を抽出
    df['weekday'] = df['date'].dt.weekday  # 曜日を抽出（0=月曜）
    df['week'] = df['date'].dt.isocalendar().week  # 週番号を抽出
    months = sorted(df['month'].unique())  # 対象月リスト
    figs = []  # 各月のグラフ格納リスト
    for month in months:  # 月ごとに処理
        month_data = df[df['month'] == month]  # 対象月のデータ抽出
        pivot = month_data.pivot(index='week', columns='weekday', values='probability')  # ヒートマップ用データ
        days = month_data.pivot(index='week', columns='weekday', values='day')  # 日付表示用
        pivot = pivot.iloc[::-1]  # 上下反転（カレンダー表示用）
        days = days.iloc[::-1]
        # カスタムカラースケール（グラデーション）
        custom_colorscale = [
            [0.0, prob_to_color(0.0)],
            [0.5, prob_to_color(0.5)],
            [1.0, prob_to_color(1.0)]
        ]
        annotations = []  # アノテーション（各日付・確率表示）
        for i, week in enumerate(pivot.index):  # 週ごと
            for j, wd in enumerate(pivot.columns):  # 曜日ごと
                val = pivot.iloc[i, j]  # 確率値
                day = days.iloc[i, j]   # 日付
                if not pd.isna(val):  # データが存在する場合
                    font_color = 'black' if val < 0.7 else 'white'  # 高確率は白文字
                    font_weight = 'bold' if val >= 0.7 else 'normal'  # 高確率は太字
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=f"<b>{int(day)}</b><br><span style='font-size:13px;'>{val:.2f}</span>",  # 日付と確率
                            showarrow=False,
                            font=dict(size=14, color=font_color, family='sans-serif'),
                            align='center',
                            bgcolor=prob_to_color(val),
                            opacity=0.8
                        )
                    )
        # Plotlyヒートマップ作成
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,  # 確率値
            x=['月','火','水','木','金','土','日'],  # 曜日ラベル
            colorscale=custom_colorscale,  # カラースケール
            zmin=0, zmax=1,  # 色の範囲
            colorbar=dict(title='生理確率')  # カラーバー
        ))
        fig.update_layout(
            title=f"{month_data['year'].iloc[0]}年{month}月",  # タイトル
            annotations=annotations,  # アノテーション追加
            margin=dict(l=40, r=40, t=60, b=40),  # 余白
            yaxis=dict(showticklabels=False),  # y軸ラベル非表示
            font=dict(family='sans-serif', size=13)  # フォント設定
        )
        figs.append(fig)  # リストに追加
    return figs, months  # グラフと月リストを返す

# --- Streamlit UI ---

st.set_page_config(page_title="生理周期カレンダー", layout="centered")  # ページ設定
st.title("生理周期カレンダー")  # タイトル表示

with st.form("cycle_form"):  # 入力フォーム
    cycle_data = st.text_input("生理周期データ（カンマ区切り）", "23,24,25,28,25")  # 周期データ入力
    start_date = st.date_input("最後の生理が始まった日", datetime(2025, 1, 1))  # 開始日入力
    submitted = st.form_submit_button("送信")  # 送信ボタン

if submitted or cycle_data:  # 入力があれば処理
    try:
        cycle_list = np.array([int(x) for x in cycle_data.split(",")])  # 入力値を数値配列に変換
        mean_cycle = np.mean(cycle_list)  # 平均周期
        std_cycle = np.std(cycle_list, ddof=1)  # 標準偏差
        if std_cycle == 0:
            std_cycle = 0.5  # 分散0の場合は0.5に置き換え
        variance_cycle = np.var(cycle_list, ddof=1)  # 分散
        st.markdown(f"**平均周期:** {mean_cycle:.2f} 日  ")  # 平均周期表示
        st.markdown(f"**バラツキ（標準偏差）:** {std_cycle:.2f} 日  ")  # 標準偏差表示
        start_date_dt = pd.to_datetime(start_date)  # 日付型に変換
        end_date = start_date_dt + timedelta(days=90)  # 90日後まで計算
        dates = pd.date_range(start_date_dt, end_date)  # 日付リスト作成
        probabilities = []  # 確率リスト
        for date in dates:  # 各日付ごとに確率計算
            days_since_last = (date - start_date_dt).days  # 最終生理からの日数
            if days_since_last == 0:
                prob = 1.0  # 初日は必ず生理
            else:
                prob = calculate_menstrual_period_probability(days_since_last, mean_cycle, std_cycle)  # 確率計算
            probabilities.append(prob)  # リストに追加
        df = pd.DataFrame({
            'date': dates,
            'probability': probabilities,
            'days_since_last': [(date - start_date_dt).days for date in dates]
        })  # データフレーム化
        st.markdown("### ヒートマップカレンダー")  # セクションタイトル
        figs, months = create_calendar_heatmap_plotly(df)  # ヒートマップ作成
        tab_labels = [f"{start_date_dt.year if m >= start_date_dt.month else start_date_dt.year+1}年{m}月" for m in months]  # タブ名
        tabs = st.tabs(tab_labels)  # タブ作成
        for i, tab in enumerate(tabs):  # 各月ごとにタブ表示
            with tab:
                st.plotly_chart(figs[i], use_container_width=True)  # ヒートマップ表示
    except Exception as e:
        st.error(f"エラー: {e}")  # エラー