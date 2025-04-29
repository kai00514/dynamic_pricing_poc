import joblib
from prophet import Prophet
from utils import load_data
import os

if __name__=='__main__':
    # モデル保存用のディレクトリを確認
    if not os.path.exists('models'):
        os.makedirs('models')
        
    df = load_data()
    for seat in df['seat_type'].unique():
        # ベースラインモデル（価格を考慮しない）
        seat_df = df[df['seat_type']==seat][['date','reservations','event_flag']]
        seat_df = seat_df.rename(columns={'date':'ds','reservations':'y'})
        m_base = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m_base.add_regressor('event_flag')
        m_base.fit(seat_df)
        joblib.dump(m_base, f'models/model_baseline_{seat}.pkl')
        
        # 価格反応モデル
        seat_df_price = df[df['seat_type']==seat][['date','reservations','event_flag','price']]
        seat_df_price = seat_df_price.rename(columns={'date':'ds','reservations':'y'})
        m_price = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m_price.add_regressor('event_flag')
        m_price.add_regressor('price')
        m_price.fit(seat_df_price)
        joblib.dump(m_price, f'models/model_price_{seat}.pkl')
        
    print("モデルの学習が完了しました。")
