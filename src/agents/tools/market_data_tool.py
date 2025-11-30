from __future__ import annotations

import datetime as dt
from typing import List, Dict

import yfinance as yf
from crewai.tools import tool


def _fetch_single_ticker_market_data(ticker: str, lookback_days: int = 180) -> Dict:
    """
    Pull a simple market stats bundle for a single ticker using yfinance.

    Returns a dict with:
        - last_price
        - period_return_pct      (return over the lookback window, in %)
        - volatility_pct         (annualized, in %)
        - max_drawdown_pct       (worst drawdown over the window, in %)
        - ma_fast_20             (20-day moving average)
        - ma_slow_60             (60-day moving average)
        - trend_label            ("uptrend", "downtrend", or "sideways / mixed")
        - sharpe_like            (rough Sharpe-style signal, rf ≈ 0)
        - n_points               (count of price points used)
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' column in price data for {ticker}")

    close_series = df["Close"].dropna()
    if close_series.empty:
        raise ValueError(f"No valid close prices for {ticker}")

    last_price = float(close_series.iloc[-1])

    # Simple total return over the lookback window
    period_return = (last_price / float(close_series.iloc[0]) - 1.0) * 100.0

    # Daily returns & volatility
    daily_returns = close_series.pct_change().dropna()
    if daily_returns.empty:
        volatility = 0.0
        sharpe_like = 0.0
    else:
        daily_std = float(daily_returns.std())
        daily_mean = float(daily_returns.mean())
        # Annualize daily std dev (252 trading days), then convert to percent
        volatility = daily_std * (252 ** 0.5) * 100.0
        sharpe_like = (
            (daily_mean / daily_std) * (252 ** 0.5) if daily_std > 0 else 0.0
        )

    # Max drawdown over the window
    rolling_max = close_series.cummax()
    drawdowns = close_series / rolling_max - 1.0
    max_drawdown = float(drawdowns.min()) * 100.0  # negative, e.g. -35%

    # Moving averages (fallbacks if the window is short)
    if len(close_series) >= 20:
        ma_fast_20 = float(close_series.tail(20).mean())
    else:
        ma_fast_20 = float(close_series.mean())

    if len(close_series) >= 60:
        ma_slow_60 = float(close_series.tail(60).mean())
    else:
        ma_slow_60 = ma_fast_20

    # Very simple trend label based on price vs MAs
    if last_price > ma_fast_20 and ma_fast_20 >= ma_slow_60:
        trend_label = "uptrend"
    elif last_price < ma_fast_20 and ma_fast_20 <= ma_slow_60:
        trend_label = "downtrend"
    else:
        trend_label = "sideways / mixed"

    return {
        "last_price": last_price,
        "period_return_pct": period_return,
        "volatility_pct": volatility,
        "max_drawdown_pct": max_drawdown,
        "ma_fast_20": ma_fast_20,
        "ma_slow_60": ma_slow_60,
        "trend_label": trend_label,
        "sharpe_like": sharpe_like,
        "n_points": int(len(close_series)),
    }


def _format_market_block(ticker: str, stats: Dict) -> str:
    """
    Turn the stats dict into a markdown block the research agents can read easily.
    """
    lp = stats["last_price"]
    ret = stats["period_return_pct"]
    vol = stats["volatility_pct"]
    dd = stats["max_drawdown_pct"]
    ma_fast = stats["ma_fast_20"]
    ma_slow = stats["ma_slow_60"]
    trend = stats["trend_label"]
    sharpe = stats["sharpe_like"]
    n = stats["n_points"]

    direction = "up" if ret >= 0 else "down"

    return (
        f"### {ticker}\n"
        f"- Last price: **${lp:,.2f}**\n"
        f"- Period return (~{n} points): **{ret:+.1f}%** ({direction})\n"
        f"- Approx. annualized volatility: **{vol:.1f}%**\n"
        f"- Max drawdown over window: **{dd:.1f}%**\n"
        f"- 20-day MA: **${ma_fast:,.2f}**, 60-day MA: **${ma_slow:,.2f}**\n"
        f"- Simple trend signal: **{trend}**\n"
        f"- Rough Sharpe-like ratio (rf≈0): **{sharpe:.2f}**\n"
    )


@tool("market_data_tool")
def market_data_tool(tickers: str, lookback_days: int = 180) -> str:
    """
    Fetch market data for a comma-separated list of tickers and return
    a markdown snapshot the research agent can plug straight into prompts.

    Args:
        tickers:
            Comma-separated tickers (e.g. "AAPL, MSFT, GOOGL").
        lookback_days:
            How many calendar days of history to pull (default: 180).

    Returns:
        Markdown summarizing recent market behavior per ticker:
        returns, volatility, max drawdown, moving averages, trend signal,
        and a simple Sharpe-like ratio.
    """
    symbols: List[str] = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        return "No valid tickers were provided to market_data_tool."

    header = (
        f"## Market Data Snapshot (last {lookback_days} days)\n"
        "_Source: yfinance daily adjusted close data_\n\n"
    )

    blocks: List[str] = []
    for sym in symbols:
        try:
            stats = _fetch_single_ticker_market_data(sym, lookback_days=lookback_days)
            blocks.append(_format_market_block(sym, stats))
        except Exception as e:
            blocks.append(
                f"### {sym}\n"
                f"- Error fetching market data: `{e}`\n"
            )

    return header + "\n".join(blocks)
