from sklearn.metrics import mean_absolute_error
import pandas as pd, numpy as np, joblib
from utils import load_data

def evaluate(seat, model_path, target='reservations'):
    df = load_data()
    seat_df = df[df['seat_type']==seat].sort_values('date')
    split = int(len(seat_df)*0.8)
    train, test = seat_df[:split], seat_df[split:]
    m = joblib.load(model_path)
    future = test.rename(columns={'date':'ds'})[['ds','price','event_flag']]
    # price モデルなら price も、baseline なら price, ignore regressors
    forecast = m.predict(future)
    y_true = test['reservations'].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{seat} MAE: {mae:.1f}')
