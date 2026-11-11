import datetime
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from trading_strategy import (
    get_stock_data,
    calculate_moving_averages,
    identify_golden_cross,
    run_single_stock_strategy,
    run_portfolio_strategy,
)
from utils import calculate_max_drawdown, calculate_sharpe_ratio

# Page Configuration
st.set_page_config(
    page_title="FinTech Multi-Stock Portfolio & AI Strategy Platform",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Stock Trading Strategy & AI Analytics Platform")
st.caption("PD2 Idea 1: Enhanced FinTech Tool for Non-Professional Investors")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Backtest Configurations")

ticker_input = st.sidebar.text_input(
    "Stock Ticker(s) (comma-separated, max 20):",
    value="MSFT, AAPL, NVDA"
)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if len(tickers) > 20:
    st.sidebar.warning("⚠️ Maximum 20 stocks allowed. Truncating list to first 20.")
    tickers = tickers[:20]

strategy_choice = st.sidebar.selectbox(
    "Select Trading Strategy:",
    options=["Golden Cross", "Death Cross", "Momentum"],
    index=0
)
strategy_map = {
    "Golden Cross": "golden",
    "Death Cross": "death",
    "Momentum": "momentum"
}
selected_strategy_key = strategy_map[strategy_choice]

# MA Window configurations
col_ma1, col_ma2 = st.sidebar.columns(2)
with col_ma1:
    short_win = st.number_input("Short MA Window", min_value=5, max_value=100, value=50, step=5)
with col_ma2:
    long_win = st.number_input("Long MA Window", min_value=20, max_value=300, value=200, step=10)

if short_win >= long_win:
    st.sidebar.error("Short MA window must be strictly smaller than Long MA window!")

# Date range selection
default_start = datetime.date(2021, 1, 1)
default_end = datetime.date(2026, 1, 1)
date_range = st.sidebar.date_input("Date Range:", value=(default_start, default_end))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = date_range
else:
    start_dt, end_dt = default_start, default_end

run_button = st.sidebar.button("🚀 Run Backtest", type="primary")

# Navigation Radio
page = st.sidebar.radio(
    "Navigate Views:",
    ("Price & Signal Chart", "Portfolio Statistics", "Detailed Trades", "🤖 AI Strategy Analysis")
)

# --- BACKTEST DATA EXECUTION & CACHING ---
@st.cache_data(ttl=1800)
def execute_backtest(ticker_list, start_d, end_d, strat, s_win, l_win):
    if len(ticker_list) == 1:
        raw_df = get_stock_data(ticker_list[0], period="5y")
        trades = run_single_stock_strategy(
            raw_df, strat, short_window=int(s_win), long_window=int(l_win)
        )
        trades.insert(0, "Ticker", ticker_list[0])
        summary = {"average_cumulative_return_pct": trades["ProfitPct"].sum() if not trades.empty else 0.0}
        stats_df = pd.DataFrame([{
            "Ticker": ticker_list[0],
            "TotalTrades": len(trades),
            "WinningTrades": int((trades["ProfitPct"] > 0).sum()) if not trades.empty else 0,
            "LosingTrades": len(trades) - int((trades["ProfitPct"] > 0).sum()) if not trades.empty else 0,
            "WinRatePct": (trades["ProfitPct"] > 0).mean() * 100 if not trades.empty else 0.0,
            "CumulativeReturnPct": trades["ProfitPct"].sum() if not trades.empty else 0.0,
            "AverageProfitPct": trades["ProfitPct"].mean() if not trades.empty else 0.0,
            "AverageHoldingDays": trades["HoldingDays"].mean() if not trades.empty else 0.0
        }])
        return summary, stats_df, trades, {ticker_list[0]: raw_df}
    else:
        summary, stats_df, trades = run_portfolio_strategy(
            ticker_list,
            start_date=str(start_d),
            end_date=str(end_d),
            strategy_name=strat,
            short_window=int(s_win),
            long_window=int(l_win)
        )
        dict_raw = {}
        for tk in ticker_list:
            try:
                dict_raw[tk] = get_stock_data(tk, period="5y")
            except Exception:
                pass
        return summary, stats_df, trades, dict_raw

# Run computation
with st.spinner("Fetching market data and calculating signals..."):
    try:
        summary_res, stats_res, trades_res, raw_dict = execute_backtest(
            tickers, start_dt, end_dt, selected_strategy_key, short_win, long_win
        )
    except Exception as e:
        st.error(f"Failed to execute backtest: {e}")
        st.stop()

# --- PAGE RENDERING ---

if page == "Price & Signal Chart":
    st.subheader("📌 Stock Price, Moving Averages & Trade Markers")
    
    selected_chart_ticker = st.selectbox("Select Ticker to Display Chart:", options=tickers)
    
    if selected_chart_ticker in raw_dict and not raw_dict[selected_chart_ticker].empty:
        df_chart = raw_dict[selected_chart_ticker].copy()
        df_chart['ShortMA'] = df_chart['Close'].rolling(window=int(short_win)).mean()
        df_chart['LongMA'] = df_chart['Close'].rolling(window=int(long_win)).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], mode='lines', name='Close Price', line=dict(color='#2563eb')))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ShortMA'], mode='lines', name=f'{short_win}-Day MA', line=dict(color='#f59e0b')))
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['LongMA'], mode='lines', name=f'{long_win}-Day MA', line=dict(color='#8b5cf6')))
        
        # Add Trade markers
        ticker_trades = trades_res[trades_res['Ticker'] == selected_chart_ticker] if not trades_res.empty else pd.DataFrame()
        if not ticker_trades.empty:
            fig.add_trace(go.Scatter(
                x=ticker_trades['BuyDate'], y=ticker_trades['BuyPrice'],
                mode='markers', name='Buy Signal',
                marker=dict(color='#16a34a', size=11, symbol='circle'),
                hovertemplate="Buy<br>%{x}<br>$%{y:.2f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=ticker_trades['SellDate'], y=ticker_trades['SellPrice'],
                mode='markers', name='Sell Signal',
                marker=dict(color='#dc2626', size=11, symbol='circle'),
                hovertemplate="Sell<br>%{x}<br>$%{y:.2f}<extra></extra>"
            ))
            
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No price data available for {selected_chart_ticker}.")

elif page == "Portfolio Statistics":
    st.subheader("📊 Portfolio & Risk Statistics Summary")
    
    if trades_res.empty:
        st.info("No trades executed across the selected stocks and parameters.")
    else:
        total_portfolio_trades = len(trades_res)
        portfolio_win_rate = (trades_res['ProfitPct'] > 0).mean() * 100
        avg_trade_profit = trades_res['ProfitPct'].mean()
        
        # Risk Metrics calculation via utils.py
        returns_series = trades_res['ProfitPct'] / 100.0
        mdd = calculate_max_drawdown(returns_series)
        sharpe = calculate_sharpe_ratio(returns_series)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades", total_portfolio_trades)
        m2.metric("Portfolio Win Rate", f"{portfolio_win_rate:.2f}%")
        m3.metric("Avg Profit / Trade", f"{avg_trade_profit:.2f}%")
        m4.metric("Max Drawdown", f"{mdd:.2%}")
        m5.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.write("---")
        st.subheader("Performance Breakdown by Ticker")
        st.dataframe(stats_res, use_container_width=True, hide_index=True)

elif page == "Detailed Trades":
    st.subheader("📜 Complete Executed Trade History")
    
    if trades_res.empty:
        st.info("No detailed trade logs available.")
    else:
        selected_filter_ticker = st.multiselect("Filter by Ticker:", options=tickers, default=tickers)
        filtered_trades = trades_res[trades_res['Ticker'].isin(selected_filter_ticker)]
        
        st.dataframe(
            filtered_trades,
            use_container_width=True,
            hide_index=True,
            column_config={
                "BuyPrice": st.column_config.NumberColumn(format="$%.2f"),
                "SellPrice": st.column_config.NumberColumn(format="$%.2f"),
                "ProfitPct": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

elif page == "🤖 AI Strategy Analysis":
    st.subheader("🤖 LLM-Enhanced Strategy Interpretation")
    st.caption("Powered by HKBU GenAI API Service")
    
    hkbu_key = st.text_input("Enter HKBU API Key for AI Analysis:", type="password")
    
    if st.button("Generate AI Insights", type="primary"):
        if not hkbu_key:
            st.warning("Please enter a valid HKBU API Key.")
        else:
            try:
                from llm_analysis import generate_llm_insight
                with st.spinner("Analyzing trading performance via LLM..."):
                    insight_text = generate_llm_insight(hkbu_key, stats_res, trades_res, strategy_choice)
                    st.markdown(insight_text)
            except ImportError:
                st.info("Role 3 module (`llm_analysis.py`) is being integrated. Placeholder text displayed below:")
                st.markdown("""
                > **AI Portfolio Performance Evaluation:**
                > 
                > * **Overview:** The backtest executed strategy across the target stocks.
                > * **Risk Assessment:** Risk parameters such as drawdown and win rate indicate structural stability over the backtest window.
                > 
                > *Disclaimer: AI generated analysis for informational and non-professional education purposes only. Not financial advice.*
                """)
            except Exception as ex:
                st.error(f"Error calling LLM Service: {ex}")