import numpy as np
import pandas as pd

def find_optimal_price(m_price, target_date, event_flag,
                       base_price, min_price, max_price, step=50):
    # 候補価格リスト
    price_grid = np.arange(base_price*0.5, base_price*1.5 + 1, step)
    # 予測用 DataFrame
    fut = pd.DataFrame({
        'ds': [pd.to_datetime(target_date)]*len(price_grid),
        'price': price_grid,
        'event_flag': [event_flag]*len(price_grid)
    })
    # 需要予測
    forecast = m_price.predict(fut)
    demand  = forecast['yhat'].values
    revenue = price_grid * demand
    # 最適価格算出
    idx           = np.argmax(revenue)
    opt_price     = price_grid[idx]
    # ビジネスルール適用
    opt_price = max(min(opt_price, max_price), min_price)
    opt_price = round(opt_price / step) * step
    return {
        'optimal_price':  opt_price,
        'predicted_demand': demand[idx],
        'predicted_revenue': revenue[idx],
        'price_grid':       price_grid,
        'revenue_grid':     revenue
    }
