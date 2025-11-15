import streamlit as st  # Streamlit本体
import numpy as np      # 数値計算用
import pandas as pd     # データフレーム操作用
from datetime import datetime, timedelta  # 日付操作
from scipy import stats  # 統計計算
import plotly.graph_objects as go  # グラフ描画
import plotly.graph_objects as go  # グラフ描画
import json
from streamlit_cookies_controller import CookieController

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

# ...existing code...

COOKIE_NAME = "cycle_history"

def load_cycle_history_from_cookie():
    if CookieController is None:
        return None
    try:
        ctrl = CookieController()
        raw = None
        # いくつかの getter 名を試す
        for getter in ("get", "get_cookie", "get_cookie_value"):
            if hasattr(ctrl, getter):
                raw = getattr(ctrl, getter)(COOKIE_NAME)
                if raw:
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    return json.loads(raw)
        # フォールバック: ctrl.cookies が dict なら探す
        if hasattr(ctrl, "cookies"):
            cookies = getattr(ctrl, "cookies")
            if isinstance(cookies, dict) and COOKIE_NAME in cookies:
                raw = cookies[COOKIE_NAME]
                return json.loads(raw)
    except Exception:
        pass
    return None

def save_cycle_history_to_cookie(hist, days=365):
    """hist: list、days: 有効期限（日数, デフォルト 365日）"""
    if CookieController is None:
        return False
    payload = json.dumps(hist, ensure_ascii=False)
    max_age = int(days * 24 * 3600)
    try:
        ctrl = CookieController()
        # try set(...)
        if hasattr(ctrl, "set"):
            try:
                ctrl.set(COOKIE_NAME, payload, max_age=max_age)
                return True
            except TypeError:
                try:
                    ctrl.set(COOKIE_NAME, payload, expires=max_age)
                    return True
                except TypeError:
                    try:
                        ctrl.set(COOKIE_NAME, payload)
                        return True
                    except Exception:
                        pass
        # try set_cookie(...)
        if hasattr(ctrl, "set_cookie"):
            try:
                ctrl.set_cookie(COOKIE_NAME, payload, max_age=max_age)
                return True
            except TypeError:
                try:
                    ctrl.set_cookie(COOKIE_NAME, payload, expires=max_age)
                    return True
                except TypeError:
                    try:
                        ctrl.set_cookie(COOKIE_NAME, payload)
                        return True
                    except Exception:
                        pass
        # 最終フォールバック: ctrl.cookies が dict なら直接書き込む（ライブラリ次第で効かない場合あり）
        if hasattr(ctrl, "cookies") and isinstance(getattr(ctrl, "cookies"), dict):
            ctrl.cookies[COOKIE_NAME] = payload
            return True
    except Exception:
        pass
    return False


# --- Streamlit UI ---

st.set_page_config(page_title="生理周期カレンダー", layout="centered")  # ページ設定
st.title("生理周期カレンダー")  # タイトル表示

# セッションに履歴がなければクッキーから読み込んで初期化
if 'cycle_history' not in st.session_state:
    try:
        hist = load_cycle_history_from_cookie()
    except Exception:
        hist = None
    st.session_state['cycle_history'] = hist if hist else ["23,24,25,28,25"]

# 履歴をクリックするとテキスト入力に適用するコールバック
def apply_history(entry):
    st.session_state['cycle_input'] = entry

# 履歴表示（展開可）
with st.expander("履歴（クリックで入力欄に反映）", expanded=False):
    history_list = st.session_state.get('cycle_history', [])
    if not history_list:
        st.write("履歴はありません")
    else:
        for i, entry in enumerate(history_list):
            # 小さなボタンで履歴を適用
            st.button(entry, key=f"apply_history_{i}", on_click=apply_history, args=(entry,))

# 単一のテキストボックス（履歴の適用先かつ新規入力欄）
with st.form("cycle_form"):
    default_val = st.session_state.get('cycle_input', "") or st.session_state['cycle_history'][0]
    cycle_data = st.text_input("生理周期データ（カンマ区切り）", value=default_val, key='cycle_input')
    start_date = st.date_input("最後の生理が始まった日", datetime(2025, 1, 1))
    submitted = st.form_submit_button("送信")

# フォーム送信時に履歴を更新してクッキーへ保存
if submitted:
    try:
        entered = st.session_state.get('cycle_input', "").strip()
        if entered:
            hist = st.session_state.get('cycle_history', [])
            if entered in hist:
                hist.remove(entered)
            hist.insert(0, entered)
            hist = hist[:10]  # 最大10件
            st.session_state['cycle_history'] = hist
            try:
                save_cycle_history_to_cookie(hist)
            except Exception:
                pass
    except Exception:
        pass

# 入力を取得して以降の計算処理を行う
cycle_input_value = st.session_state.get('cycle_input', "").strip()
if submitted or cycle_input_value:
    try:
        # ...existing calculation and plotting code...
        cycle_list = np.array([int(x) for x in cycle_input_value.split(",")])  # 入力値を数値配列に変換
        mean_cycle = np.mean(cycle_list)  # 平均周期
        std_cycle = np.std(cycle_list, ddof=1)  # 標準偏差
        if std_cycle == 0:
            std_cycle = 0.5
        variance_cycle = np.var(cycle_list, ddof=1)
        st.markdown(f"**平均周期:** {mean_cycle:.2f} 日  ")
        st.markdown(f"**バラツキ（標準偏差）:** {std_cycle:.2f} 日  ")
        start_date_dt = pd.to_datetime(start_date)
        end_date = start_date_dt + timedelta(days=90)
        dates = pd.date_range(start_date_dt, end_date)
        probabilities = []
        for date in dates:
            days_since_last = (date - start_date_dt).days
            if days_since_last == 0:
                prob = 1.0
            else:
                prob = calculate_menstrual_period_probability(days_since_last, mean_cycle, std_cycle)
            probabilities.append(prob)
        df = pd.DataFrame({
            'date': dates,
            'probability': probabilities,
            'days_since_last': [(date - start_date_dt).days for date in dates]
        })
        st.markdown("### ヒートマップカレンダー")
        figs, months = create_calendar_heatmap_plotly(df)
        tab_labels = [f"{start_date_dt.year if m >= start_date_dt.month else start_date_dt.year+1}年{m}月" for m in months]
        tabs = st.tabs(tab_labels)
        for i, tab in enumerate(tabs):
            with tab:
                st.plotly_chart(figs[i], use_container_width=True)
    except Exception as e:
        st.error(f"エラー: {e}")
