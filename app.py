import streamlit as st
import pandas as pd, numpy as np, joblib
from utils import load_data
from datetime import datetime

st.title("ネットカフェ動的価格提案 PoC")

# デバッグ情報の追加
st.sidebar.subheader("デバッグ情報")

# データ読み込み
try:
    df = load_data()
    st.sidebar.success("データの読み込みに成功しました")
    st.sidebar.text(f"データ件数: {len(df)}行")
    st.sidebar.text(f"席種: {', '.join(df['seat_type'].unique())}")
except Exception as e:
    st.sidebar.error(f"データ読み込みエラー: {e}")
    st.stop()

# --- 1. 入力 UI ---
date = st.date_input("日付を選択")

# 将来日時かどうかを判定
date_pd = pd.to_datetime(date)
today = pd.to_datetime(datetime.now().date())
is_future_date = date_pd > today

# 選択した日付の情報表示
st.sidebar.text(f"選択日付: {date}")
st.sidebar.text(f"将来の日付: {'はい' if is_future_date else 'いいえ'}")

# 利用可能な席種を表示
available_seats = df['seat_type'].unique()
seat = st.selectbox("席種を選択", available_seats)

# --- 2. データとモデル読み込み ---
date_pd = pd.to_datetime(date)
today = pd.to_datetime(datetime.now().date())

# データ存在チェック（過去の日付のみ）
if not is_future_date:
    date_exists = date_pd in df['date'].values
    if not date_exists:
        st.error(f"選択された日付 {date} のデータが存在しません。別の日付を選択してください。")
        st.stop()

# event_flagの決定（将来日の場合はユーザーに入力を促す）
if is_future_date:
    st.subheader("将来の日付が選択されました")
    event_flag = st.radio("イベントの予定はありますか？", [0, 1], 
                         format_func=lambda x: "あり（人気度70%以上）" if x == 1 else "なし（または人気度70%未満）")
    
    # 将来の日付の基準価格をユーザーに入力してもらう
    default_prices = {
        'vip': 15000,
        'private': 8000,
        'standard': 5000,
        'open': 3000
    }
    default_price = default_prices.get(seat, 5000)
    base_price = st.number_input(f"基準価格（円）", min_value=1000, max_value=50000, value=default_price, step=100)
else:
    # 過去データの場合は通常通り取得
    event_flag_values = df[df['date'] == date_pd]['event_flag']
    event_flag = int(event_flag_values.max()) if len(event_flag_values) > 0 else 0
    
    price_data = df[(df['date'] == date_pd) & (df['seat_type'] == seat)]['price']
    base_price = int(price_data.iloc[0]) if len(price_data) > 0 else default_prices.get(seat, 5000)

# モデル読み込み
try:
    m_base = joblib.load(f'models/model_baseline_{seat}.pkl')
    m_price = joblib.load(f'models/model_price_{seat}.pkl')
except FileNotFoundError:
    st.error(f"モデルファイルが見つかりません。train.pyを実行してモデルを学習してください。")
    st.stop()

# --- 3. 需要予測表示 ---
future_df = pd.DataFrame({'ds':[date_pd], 'event_flag':[event_flag]})
pred_base = m_base.predict(future_df)['yhat'].iloc[0]
st.metric("予測予約件数 (ベースライン)", f"{pred_base:.0f} 件")

# --- 4. 最適価格計算 ---
price_grid = np.arange(base_price*0.5, base_price*1.5, 50)
fut = pd.DataFrame({
    'ds': [date_pd]*len(price_grid),
    'price': price_grid,
    'event_flag': [event_flag]*len(price_grid)
})
fc = m_price.predict(fut)
revenue = price_grid * fc['yhat'].values
opt_idx = revenue.argmax()
opt_price = price_grid[opt_idx]
opt_demand = fc['yhat'].values[opt_idx]
st.metric("推奨価格", f"{opt_price:.0f} 円")
st.metric("推奨予約件数", f"{opt_demand:.0f} 件")
st.metric("予測売上", f"{(opt_price*opt_demand):.0f} 円")

# --- 5. チャート表示 ---
chart_data = pd.DataFrame({
    'price': price_grid,
    'pred_revenue': revenue
}).set_index('price')
st.line_chart(chart_data)

# --- 6. 管理者承認・修正 ---
new_price = st.slider("最終価格を調整", float(base_price*0.5), float(base_price*1.5), float(opt_price), step=50.0)
if st.button("価格を確定"):
    st.success(f"{date} の {seat} 席の価格を {new_price:.0f} 円に設定しました。")
