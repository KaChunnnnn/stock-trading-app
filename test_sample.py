from trading_strategy import run_portfolio_strategy
from datetime import date
import json
import pandas as pd

def get_sample_backtest():
    portfolio_summary, stats_df, combined_trades = run_portfolio_strategy(
        ["MSFT", "AAPL"],
        start_date=date(2021, 1, 1),
        end_date=date(2026, 1, 1),
        strategy_name="golden",
        short_window=50,
        long_window=200
    )

    # 確保 trade_records 裡面的 Timestamp 物件轉換成字串，避免 JSON 序列化失敗
    if not combined_trades.empty:
        if 'BuyDate' in combined_trades.columns:
            combined_trades['BuyDate'] = combined_trades['BuyDate'].astype(str)
        if 'SellDate' in combined_trades.columns:
            combined_trades['SellDate'] = combined_trades['SellDate'].astype(str)

    sample_backtest_input = {
        "portfolio_summary": portfolio_summary,
        "stock_statistics": stats_df.to_dict("records"),
        "trade_records": combined_trades.to_dict("records")
    }
    return sample_backtest_input

sample_backtest_input = get_sample_backtest()

if __name__ == "__main__":
    print(json.dumps(sample_backtest_input, indent=2, ensure_ascii=False))