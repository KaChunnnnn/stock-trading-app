import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
from datetime import datetime, timedelta
from utils import calculate_max_drawdown, calculate_sharpe_ratio

# Download 5 years of MSFT data
def get_stock_data(ticker, period="5y"):
    """Download historical price data for a ticker.

    Args:
        ticker: Yahoo Finance ticker symbol.
        period: Yahoo Finance history period, such as ``"5y"``.

    Returns:
        A DataFrame containing the downloaded price history.

    Raises:
        ValueError: If Yahoo Finance returns no usable price data.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if (
        data is None
        or data.empty
        or 'Close' not in data.columns
        or data['Close'].dropna().empty
    ):
        raise ValueError(f"No usable Yahoo Finance data was returned for {ticker}.")
    return data

# Calculate moving averages
def calculate_moving_averages(data):
    """Add 50-day and 200-day closing-price averages to price data.

    Args:
        data: Price DataFrame containing a ``Close`` column.

    Returns:
        The input DataFrame with ``MA50`` and ``MA200`` columns added.
    """
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

# Identify golden cross (buy signals)
def identify_golden_cross(data):
    """Add the golden-cross indicator to moving-average price data.

    Args:
        data: DataFrame containing ``MA50`` and ``MA200`` columns.

    Returns:
        The input DataFrame with ``Signal`` and ``GoldenCross`` columns added.
    """
    data['Signal'] = 0  # Initialize signal column with 0
    # Golden Cross occurs when MA50 crosses above MA200
    data['GoldenCross'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
    return data

# Implement trading strategy
def implement_strategy(data):
    """Backtest the existing golden-cross strategy.

    Args:
        data: DataFrame containing ``Close`` and ``GoldenCross`` columns.

    Returns:
        Trade records using the standard ``TRADE_COLUMNS`` schema.
    """
    positions = []

    # Need at least 200 days to calculate the 200-day MA
    data = data.iloc[200:].copy()

    buy_dates = data[data['GoldenCross'] == True].index.tolist()

    for buy_date in buy_dates:
        # Get buy price
        buy_price = data.loc[buy_date, 'Close']

        # Calculate target sell price (15% profit)
        target_price = buy_price * 1.15

        # Set maximum holding period
        max_sell_date = buy_date + pd.Timedelta(days=60)

        # Get data slice for potential sell period
        sell_period = data.loc[buy_date:max_sell_date].copy()

        # Check if target price is reached during the period
        target_reached = sell_period[sell_period['Close'] >= target_price]

        if not target_reached.empty:
            # Sell at first date target is reached
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            # Sell at end of maximum holding period
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                # Skip if no valid sell date (should not happen in practice)
                continue

        # Calculate holding period in calendar days
        holding_days = (sell_date - buy_date).days

        # Calculate profit
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions, columns=TRADE_COLUMNS)


# Keep the trade result schema consistent across all strategies.
TRADE_COLUMNS = [
    'BuyDate',
    'BuyPrice',
    'SellDate',
    'SellPrice',
    'HoldingDays',
    'ProfitPct',
    'SellReason',
]


def _positions_from_signals(data, buy_signal, sell_signal, sell_reason):
    """Build non-overlapping trades from entry and exit signal series."""
    positions = []
    buy_date = None

    for date in data.index:
        if buy_date is None:
            if bool(buy_signal.loc[date]):
                buy_date = date
            continue

        if bool(sell_signal.loc[date]):
            buy_price = data.loc[buy_date, 'Close']
            sell_price = data.loc[date, 'Close']
            positions.append({
                'BuyDate': buy_date,
                'BuyPrice': buy_price,
                'SellDate': date,
                'SellPrice': sell_price,
                'HoldingDays': (date - buy_date).days,
                'ProfitPct': (sell_price / buy_price - 1) * 100,
                'SellReason': sell_reason,
            })
            buy_date = None

    # Close any remaining position at the last available close.
    if buy_date is not None:
        sell_date = data.index[-1]
        buy_price = data.loc[buy_date, 'Close']
        sell_price = data.loc[sell_date, 'Close']
        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': (sell_date - buy_date).days,
            'ProfitPct': (sell_price / buy_price - 1) * 100,
            'SellReason': 'End of data',
        })

    return pd.DataFrame(positions, columns=TRADE_COLUMNS)


def implement_strategy_deathcross(data, short_window=50, long_window=200):
    """Backtest a short/long moving-average death-cross strategy.

    Args:
        data: Price DataFrame containing a ``Close`` column.
        short_window: Number of observations for the short moving average.
        long_window: Number of observations for the long moving average.

    Returns:
        Trade records using the standard ``TRADE_COLUMNS`` schema.
    """
    strategy_data = data.copy()
    strategy_data['ShortMA'] = strategy_data['Close'].rolling(
        window=short_window
    ).mean()
    strategy_data['LongMA'] = strategy_data['Close'].rolling(
        window=long_window
    ).mean()
    death_cross = (
        (strategy_data['ShortMA'] < strategy_data['LongMA'])
        & (strategy_data['ShortMA'].shift(1) >= strategy_data['LongMA'].shift(1))
    )
    buy_signal = strategy_data['ShortMA'] > strategy_data['LongMA']

    return _positions_from_signals(
        strategy_data,
        buy_signal.fillna(False),
        death_cross.fillna(False),
        'Death cross',
    )


def implement_strategy_momentum(data, momentum_window=20):
    """Backtest a price-momentum strategy.

    Args:
        data: Price DataFrame containing a ``Close`` column.
        momentum_window: Number of observations used to calculate momentum.

    Returns:
        Trade records using the standard ``TRADE_COLUMNS`` schema.
    """
    strategy_data = data.copy()
    momentum = strategy_data['Close'].pct_change(periods=momentum_window)
    buy_signal = momentum > 0
    sell_signal = momentum <= 0

    return _positions_from_signals(
        strategy_data,
        buy_signal.fillna(False),
        sell_signal.fillna(False),
        'Momentum <= 0',
    )


def run_single_stock_strategy(
    df,
    strategy_name,
    short_window=50,
    long_window=200,
):
    """Backtest one stock with a golden, death-cross, or momentum strategy.

    Args:
        df: Price history DataFrame containing a ``Close`` column.
        strategy_name: Strategy identifier: ``"golden"``, ``"death"``, or
            ``"momentum"``.
        short_window: Short moving-average window for crossover strategies.
        long_window: Long moving-average window for crossover strategies.

    Returns:
        A DataFrame with exactly the columns in ``TRADE_COLUMNS``.

    Raises:
        ValueError: If the strategy, input data, or window sizes are invalid.
    """
    if not isinstance(strategy_name, str) or strategy_name.lower() not in (
        'golden', 'death', 'momentum'
    ):
        raise ValueError(
            "strategy_name must be one of: 'golden', 'death', or 'momentum'."
        )
    if (
        not isinstance(short_window, int)
        or isinstance(short_window, bool)
        or not isinstance(long_window, int)
        or isinstance(long_window, bool)
        or short_window <= 0
        or long_window <= 0
    ):
        raise ValueError("Moving-average windows must be positive integers.")
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    if 'Close' not in df.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    strategy_data = df.copy()
    strategy_data = strategy_data.dropna(subset=['Close'])
    if strategy_data.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    strategy_name = strategy_name.lower()

    if strategy_name == 'golden':
        # Map caller-selected windows to the column names used by the existing functions.
        strategy_data['MA50'] = strategy_data['Close'].rolling(
            window=short_window
        ).mean()
        strategy_data['MA200'] = strategy_data['Close'].rolling(
            window=long_window
        ).mean()
        strategy_data = identify_golden_cross(strategy_data)
        positions = implement_strategy(strategy_data)
    elif strategy_name == 'death':
        positions = implement_strategy_deathcross(
            strategy_data,
            short_window=short_window,
            long_window=long_window,
        )
    elif strategy_name == 'momentum':
        # Momentum uses a fixed 20-day lookback; MA windows are not used.
        positions = implement_strategy_momentum(strategy_data, momentum_window=20)
    return positions.reindex(columns=TRADE_COLUMNS)


# Analyze the results
def analyze_results(positions):
    """Print trade performance and return drawdown and Sharpe statistics.

    Args:
        positions: Trade DataFrame containing profit and holding-period fields.

    Returns:
        A message for empty input, otherwise a dictionary with maximum
        drawdown and Sharpe ratio.
    """
    if positions.empty:
        return "No trading signals detected"

    # Summary statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0

    avg_holding = positions['HoldingDays'].mean()

    target_reached = len(positions[positions['SellReason'] == 'Target reached'])
    max_period = len(positions[positions['SellReason'] == 'Max holding period'])

    print("\n===== Trading Strategy Results (Golden Cross)=====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {loss_trades}")
    print(f"Average Profit: {avg_profit:.2f}%")

    df = positions.copy()
    if "strategy_return" not in df.columns:
        df["strategy_return"] = df["ProfitPct"] / 100
    max_drawdown = calculate_max_drawdown(df["strategy_return"])
    sharpe_ratio = calculate_sharpe_ratio(df["strategy_return"])
    stats = {
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
    }

    return stats

# Main function
def main():
    """Run and return the default MSFT golden-cross backtest.

    Returns:
        A DataFrame containing the simulated trades.
    """
    # Get stock data
    data = get_stock_data("MSFT")

    # Calculate moving averages
    data = calculate_moving_averages(data)

    # Identify golden cross
    data = identify_golden_cross(data)

    # Implement strategy
    positions = implement_strategy(data)

    # Analyze results
    analyze_results(positions)

    # Return the detailed positions dataframe
    return positions


PORTFOLIO_STATS_COLUMNS = [
    'Ticker',
    'TotalTrades',
    'WinningTrades',
    'LosingTrades',
    'WinRatePct',
    'CumulativeReturnPct',
    'AverageProfitPct',
    'AverageHoldingDays',
]


def run_portfolio_strategy(
    ticker_list,
    start_date,
    end_date,
    strategy_name,
    short_window=50,
    long_window=200,
):
    """Backtest a strategy for up to 20 tickers over a date range.

    Args:
        ticker_list: Iterable of Yahoo Finance ticker symbols.
        start_date: Inclusive start date accepted by Yahoo Finance.
        end_date: Exclusive end date accepted by Yahoo Finance.
        strategy_name: Strategy identifier: ``"golden"``, ``"death"``, or
            ``"momentum"``.
        short_window: Short moving-average window for crossover strategies.
        long_window: Long moving-average window for crossover strategies.

    Returns:
        A tuple of portfolio summary dictionary, per-stock statistics
        DataFrame, and combined trade DataFrame (including ticker labels).

    Raises:
        ValueError: If more than 20 tickers or an unsupported strategy is given.
    """
    try:
        tickers = list(ticker_list)
    except TypeError as error:
        raise ValueError("ticker_list must be an iterable of ticker symbols.") from error
    if len(tickers) > 20:
        raise ValueError("A portfolio can contain no more than 20 stocks.")
    if not isinstance(strategy_name, str) or strategy_name.lower() not in (
        "golden", "death", "momentum"
    ):
        raise ValueError(
            "strategy_name must be one of: 'golden', 'death', or 'momentum'."
        )

    stock_stats = []
    trade_records = []

    for ticker in tickers:
        try:
            try:
                # Fetch the requested date range for this ticker.
                data = yf.Ticker(ticker).history(
                    start=start_date,
                    end=end_date,
                )
            finally:
                # Pause after each request to reduce the chance of rate limiting.
                time.sleep(0.4)

            if (
                data is None
                or not isinstance(data, pd.DataFrame)
                or data.empty
                or 'Close' not in data.columns
                or data['Close'].dropna().empty
            ):
                warnings.warn(
                    f"Skipping {ticker}: no usable stock data was returned.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            positions = run_single_stock_strategy(
                data,
                strategy_name,
                short_window=short_window,
                long_window=long_window,
            )
        except Exception as error:
            warnings.warn(
                f"Skipping {ticker}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        total_trades = len(positions)
        winning_trades = int((positions['ProfitPct'] > 0).sum())
        losing_trades = total_trades - winning_trades
        cumulative_return_pct = (
            ((1 + positions['ProfitPct'] / 100).prod() - 1) * 100
            if total_trades
            else 0.0
        )
        average_profit_pct = (
            float(positions['ProfitPct'].mean()) if total_trades else 0.0
        )
        average_holding_days = (
            float(positions['HoldingDays'].mean()) if total_trades else 0.0
        )
        win_rate_pct = (
            winning_trades / total_trades * 100 if total_trades else 0.0
        )

        stock_stats.append({
            'Ticker': ticker,
            'TotalTrades': total_trades,
            'WinningTrades': winning_trades,
            'LosingTrades': losing_trades,
            'WinRatePct': win_rate_pct,
            'CumulativeReturnPct': cumulative_return_pct,
            'AverageProfitPct': average_profit_pct,
            'AverageHoldingDays': average_holding_days,
        })

        labeled_positions = positions.copy()
        labeled_positions.insert(0, 'Ticker', ticker)
        trade_records.append(labeled_positions)

    stats_df = pd.DataFrame(stock_stats, columns=PORTFOLIO_STATS_COLUMNS)
    trade_columns = ['Ticker'] + TRADE_COLUMNS
    combined_trades = (
        pd.concat(trade_records, ignore_index=True)
        if trade_records
        else pd.DataFrame(columns=trade_columns)
    )
    portfolio_summary = {
        'average_cumulative_return_pct': (
            float(stats_df['CumulativeReturnPct'].mean())
            if not stats_df.empty
            else 0.0
        ),
        'successful_stock_count': len(stats_df),
    }

    return portfolio_summary, stats_df, combined_trades


if __name__ == "__main__":
    try:
        stock_data = get_stock_data("MSFT")
        for selected_strategy in ("golden", "death", "momentum"):
            print(f"\n===== {selected_strategy.title()} Strategy =====")
            try:
                strategy_positions = run_single_stock_strategy(
                    stock_data,
                    selected_strategy,
                    short_window=50,
                    long_window=200,
                )
                strategy_stats = analyze_results(strategy_positions)
                if isinstance(strategy_stats, dict):
                    print(f"Maximum Drawdown: {strategy_stats['Max Drawdown']:.2%}")
                    print(f"Sharpe Ratio: {strategy_stats['Sharpe Ratio']:.2f}")
                print("\nDetailed Trades:")
                print(strategy_positions.to_string(index=False))
            except Exception as error:
                print(f"Could not run {selected_strategy} strategy: {error}")
    except Exception as error:
        print(f"Could not download MSFT data: {error}")

    print("\n===== Portfolio Example (Golden Strategy) =====")
    try:
        portfolio_summary, stock_stats, portfolio_trades = run_portfolio_strategy(
            ["MSFT", "AAPL"],
            start_date="2021-01-01",
            end_date="2026-01-01",
            strategy_name="golden",
            short_window=50,
            long_window=200,
        )
        print("Portfolio summary:")
        print(portfolio_summary)
        print("\nIndividual stock statistics:")
        print(stock_stats.to_string(index=False))
        print("\nCombined trade records:")
        print(portfolio_trades.to_string(index=False))
    except Exception as error:
        print(f"Could not run the portfolio example: {error}")