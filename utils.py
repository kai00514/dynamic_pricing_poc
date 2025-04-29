import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """
    データを読み込み、前処理を行う関数
    
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    """
    # 各CSVファイルを読み込む
    reservations = pd.read_csv('data/reservations.csv')
    sales = pd.read_csv('data/sales.csv')
    prices = pd.read_csv('data/prices.csv')
    events = pd.read_csv('data/events.csv')
    
    # 日付をdatetime型に変換
    reservations['date'] = pd.to_datetime(reservations['date'])
    sales['date'] = pd.to_datetime(sales['date'])
    prices['date'] = pd.to_datetime(prices['date'])
    events['date'] = pd.to_datetime(events['date'])
    
    # イベントフラグを作成（イベントの人気度が70以上なら1、それ以外は0）
    events['event_flag'] = (events['popularity'] >= 70).astype(int)
    
    # 予約データとイベントデータを結合
    df = reservations.merge(events[['date', 'event_flag']], on='date', how='left')
    
    # 価格データを結合
    df = df.merge(prices[['date', 'seat_type', 'price']], on=['date', 'seat_type'], how='left')
    
    return df
