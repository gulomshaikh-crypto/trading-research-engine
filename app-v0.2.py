from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

FAPI = "https://fapi.binance.com"
FUTURES_DATA = "https://fapi.binance.com/futures/data"
SPOT_DATA = "https://data-api.binance.vision"
DB_PATH = Path("paper_journal.sqlite3")


class BinanceDataError(RuntimeError):
    pass


def api_get(url: str, params: dict[str, Any] | None = None, timeout: int = 10) -> Any:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        raise BinanceDataError(f"Binance request failed: {exc}") from exc
    if isinstance(payload, dict) and payload.get("code") not in (None, 200):
        raise BinanceDataError(f"Binance API error: {payload}")
    return payload


def klines(symbol: str, interval: str, limit: int = 300, *, spot: bool = False) -> pd.DataFrame:
    base = SPOT_DATA if spot else FAPI
    path = "/api/v3/klines" if spot else "/fapi/v1/klines"
    rows = api_get(f"{base}{path}", {"symbol": symbol, "interval": interval, "limit": limit})
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise BinanceDataError("No kline data returned")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def ratio_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ["longShortRatio", "longAccount", "shortAccount", "longPosition", "shortPosition"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _spot_taker_from_klines(k: pd.DataFrame) -> pd.DataFrame:
    # Spot klines include taker-buy base volume. Sell taker volume is approximated as
    # total base volume minus taker-buy base volume. This is useful for a fallback
    # order-flow proxy when Futures analytics are unavailable.
    out = pd.DataFrame({
        "buyVol": k["taker_buy_volume"].astype(float),
        "sellVol": (k["volume"] - k["taker_buy_volume"]).clip(lower=0).astype(float),
    })
    out["buySellRatio"] = out["buyVol"] / out["sellVol"].replace(0, np.nan)
    return out.fillna(1.0)


def futures_snapshot(symbol: str, interval: str) -> dict[str, Any]:
    ticker = api_get(f"{FAPI}/fapi/v1/ticker/24hr", {"symbol": symbol})
    k = klines(symbol, interval, 300, spot=False)

    oi = pd.DataFrame(api_get(f"{FUTURES_DATA}/openInterestHist", {"symbol": symbol, "period": interval, "limit": 48}))
    if not oi.empty:
        for c in ["sumOpenInterest", "sumOpenInterestValue"]:
            if c in oi.columns:
                oi[c] = pd.to_numeric(oi[c], errors="coerce")

    funding = pd.DataFrame(api_get(f"{FAPI}/fapi/v1/fundingRate", {"symbol": symbol, "limit": 12}))
    if not funding.empty and "fundingRate" in funding.columns:
        funding["fundingRate"] = pd.to_numeric(funding["fundingRate"], errors="coerce")

    global_ls = ratio_df(api_get(
        f"{FUTURES_DATA}/globalLongShortAccountRatio",
        {"symbol": symbol, "period": interval, "limit": 48},
    ))
    top_ls = ratio_df(api_get(
        f"{FUTURES_DATA}/topLongShortPositionRatio",
        {"symbol": symbol, "period": interval, "limit": 48},
    ))

    taker = pd.DataFrame(api_get(
        f"{FUTURES_DATA}/takerlongshortRatio",
        {"symbol": symbol, "period": interval, "limit": 48},
    ))
    if not taker.empty:
        for c in ["buySellRatio", "buyVol", "sellVol"]:
            if c in taker.columns:
                taker[c] = pd.to_numeric(taker[c], errors="coerce")

    book = api_get(f"{FAPI}/fapi/v1/depth", {"symbol": symbol, "limit": 100})

    return {
        "ticker": ticker, "klines": k, "oi": oi, "funding": funding,
        "global_ls": global_ls, "top_ls": top_ls, "taker": taker, "book": book,
        "source": "Binance USD-M Futures",
        "limitations": "Full derivatives layer available.",
    }


def spot_snapshot(symbol: str, interval: str) -> dict[str, Any]:
    ticker = api_get(f"{SPOT_DATA}/api/v3/ticker/24hr", {"symbol": symbol})
    k = klines(symbol, interval, 300, spot=True)
    book = api_get(f"{SPOT_DATA}/api/v3/depth", {"symbol": symbol, "limit": 100})
    return {
        "ticker": ticker,
        "klines": k,
        "oi": pd.DataFrame(),
        "funding": pd.DataFrame(),
        "global_ls": pd.DataFrame(),
        "top_ls": pd.DataFrame(),
        "taker": _spot_taker_from_klines(k).tail(48),
        "book": book,
        "source": "Binance Spot public market-data fallback",
        "limitations": "Futures endpoint unavailable from this cloud IP: OI, funding and long/short metrics are neutralized; spot taker flow is used instead.",
    }


def snapshot(symbol: str, interval: str) -> dict[str, Any]:
    try:
        return futures_snapshot(symbol, interval)
    except BinanceDataError as futures_exc:
        try:
            snap = spot_snapshot(symbol, interval)
            snap["futures_error"] = str(futures_exc)
            return snap
        except BinanceDataError as spot_exc:
            raise BinanceDataError(f"Futures failed ({futures_exc}); Spot fallback also failed ({spot_exc})") from spot_exc


def ema(series: pd.Series, period: int) -> float:
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    value = out.iloc[-1]
    return float(value if np.isfinite(value) else 50.0)


def atr(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def safe_pct(a: float, b: float) -> float:
    return (b / a - 1) * 100 if a else 0.0


def realized_vol_annualized(close: pd.Series, bars_per_day: int) -> float:
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 20:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(bars_per_day * 365) * 100)


def flow_ratio(df: pd.DataFrame, n: int) -> float:
    if df.empty or not {"buyVol", "sellVol"}.issubset(df.columns):
        return 1.0
    x = df.tail(n)
    buy = float(x["buyVol"].sum())
    sell = float(x["sellVol"].sum())
    return buy / sell if sell > 0 else 1.0


def extract_features(snap: dict[str, Any], interval: str) -> dict[str, float]:
    k = snap["klines"]
    close = k["close"]
    price = float(close.iloc[-1])
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    a14 = atr(k, 14)
    bars_per_day = 24 if interval == "1h" else 6

    oi = snap["oi"]
    oi_last = float(oi["sumOpenInterest"].iloc[-1]) if not oi.empty else 0.0
    oi_6 = float(oi["sumOpenInterest"].iloc[-min(7, len(oi))]) if not oi.empty else oi_last
    oi_24 = float(oi["sumOpenInterest"].iloc[-min(25 if interval == "1h" else 7, len(oi))]) if not oi.empty else oi_last

    funding = snap["funding"]
    fund_last = float(funding["fundingRate"].iloc[-1]) if not funding.empty else 0.0
    fund_avg = float(funding["fundingRate"].mean()) if not funding.empty else 0.0

    gls = snap["global_ls"]
    tls = snap["top_ls"]
    global_ls = float(gls["longShortRatio"].iloc[-1]) if not gls.empty else 1.0
    top_ls = float(tls["longShortRatio"].iloc[-1]) if not tls.empty else 1.0

    bids = sum(float(q) for _, q in snap["book"].get("bids", []))
    asks = sum(float(q) for _, q in snap["book"].get("asks", []))
    book_ratio = bids / asks if asks > 0 else 1.0

    lookback24 = min(bars_per_day + 1, len(close))
    lookback7d = min(bars_per_day * 7 + 1, len(close))
    p24 = float(close.iloc[-lookback24]) if lookback24 > 1 else price
    p7 = float(close.iloc[-lookback7d]) if lookback7d > 1 else price
    recent = k.tail(min(12, len(k)))

    ticker = snap["ticker"]
    return {
        "price": price,
        "change_24h_pct": float(ticker.get("priceChangePercent", safe_pct(p24, price))),
        "change_7d_pct": safe_pct(p7, price),
        "ema20": e20,
        "ema50": e50,
        "ema_spread_pct": (e20 / e50 - 1) * 100 if e50 else 0.0,
        "price_vs_ema50_pct": (price / e50 - 1) * 100 if e50 else 0.0,
        "rsi14": rsi(close, 14),
        "atr14": a14,
        "atr_pct": a14 / price * 100 if price else 0.0,
        "realized_vol_ann_pct": realized_vol_annualized(close.tail(bars_per_day * 7 + 1), bars_per_day),
        "oi_change_6bars_pct": safe_pct(oi_6, oi_last),
        "oi_change_24h_pct": safe_pct(oi_24, oi_last),
        "funding_last_pct": fund_last * 100,
        "funding_avg_pct": fund_avg * 100,
        "global_ls": global_ls,
        "top_ls": top_ls,
        "taker_ratio_6bars": flow_ratio(snap["taker"], 6),
        "taker_ratio_24h": flow_ratio(snap["taker"], min(bars_per_day, len(snap["taker"]))),
        "book_bid_ask_ratio": book_ratio,
        "recent_high": float(recent["high"].max()),
        "recent_low": float(recent["low"].min()),
    }


def detect_regime(f: dict[str, float]) -> str:
    trend_up = f["price"] > f["ema20"] > f["ema50"]
    trend_down = f["price"] < f["ema20"] < f["ema50"]
    high_vol = f["atr_pct"] > 1.2
    if trend_up and f["oi_change_24h_pct"] > 0.5:
        return "TREND_UP_HIGH_VOL" if high_vol else "TREND_UP"
    if trend_down and f["oi_change_24h_pct"] > 0.5:
        return "TREND_DOWN_HIGH_VOL" if high_vol else "TREND_DOWN"
    if high_vol:
        return "RANGE_HIGH_VOL"
    return "RANGE"


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_signal(f: dict[str, float]) -> dict[str, float | str]:
    trend = 0.0
    trend += clip(f["ema_spread_pct"] / 0.8) * 0.45
    trend += clip(f["price_vs_ema50_pct"] / 2.0) * 0.35
    trend += clip((f["rsi14"] - 50) / 25) * 0.20

    direction = 1.0 if f["change_24h_pct"] >= 0 else -1.0
    oi_component = clip(f["oi_change_24h_pct"] / 3.0) * direction
    funding_penalty = clip(abs(f["funding_last_pct"]) / 0.05)
    derivatives = clip(0.75 * oi_component - 0.25 * funding_penalty * direction)

    taker = clip(math.log(max(f["taker_ratio_6bars"], 1e-6)) / 0.25)
    book = clip(math.log(max(f["book_bid_ask_ratio"], 1e-6)) / 0.7)
    order_flow = clip(0.7 * taker + 0.3 * book)

    top = clip(math.log(max(f["top_ls"], 1e-6)) / 0.6)
    crowd = 0.0
    if f["global_ls"] > 2.0:
        crowd = -clip((f["global_ls"] - 2.0) / 2.0)
    elif f["global_ls"] < 0.6:
        crowd = clip((0.6 - f["global_ls"]) / 0.4)
    positioning = clip(0.45 * top + 0.55 * crowd)

    composite = clip(0.35 * trend + 0.25 * derivatives + 0.25 * order_flow + 0.15 * positioning)
    decision = "LONG" if composite >= 0.35 else "SHORT" if composite <= -0.35 else "WAIT"
    confidence = min(0.90, 0.50 + abs(composite) * 0.40)

    return {
        "trend": round(trend, 3),
        "derivatives": round(derivatives, 3),
        "order_flow": round(order_flow, 3),
        "positioning": round(positioning, 3),
        "composite": round(composite, 3),
        "score_100": round(composite * 100, 1),
        "decision": decision,
        "confidence": round(confidence, 3),
    }


def risk_plan(f: dict[str, float], decision: str) -> dict[str, float | str]:
    price = f["price"]
    a = max(f["atr14"], price * 0.002)
    daily_sigma = (f["realized_vol_ann_pct"] / 100) / math.sqrt(365) if f["realized_vol_ann_pct"] > 0 else 0.0
    expected_move = price * daily_sigma

    if decision == "LONG":
        entry = max(price, f["recent_high"])
        stop = entry - 1.25 * a
        target1 = entry + 1.75 * a
        target2 = entry + 2.75 * a
    elif decision == "SHORT":
        entry = min(price, f["recent_low"])
        stop = entry + 1.25 * a
        target1 = entry - 1.75 * a
        target2 = entry - 2.75 * a
    else:
        entry = price
        stop = price - 1.25 * a
        target1 = price + 1.75 * a
        target2 = price + 2.75 * a

    risk = abs(entry - stop)
    return {
        "expected_daily_move_usd": round(expected_move, 2),
        "expected_daily_move_pct": round(daily_sigma * 100, 2),
        "reference_entry": round(entry, 2),
        "invalidation": round(stop, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "rr1": round(abs(target1 - entry) / risk if risk else 0.0, 2),
        "rr2": round(abs(target2 - entry) / risk if risk else 0.0, 2),
        "note": "Reference levels only; the live score is not a trading recommendation. Validate the historical proxy below.",
    }



INTERVAL_MS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}


def historical_klines(symbol: str, interval: str, bars: int = 2000) -> pd.DataFrame:
    """Fetch closed Spot candles in chronological order, paging Binance's public market-data API."""
    bars = int(max(300, min(bars, 5000)))
    step_ms = INTERVAL_MS[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (bars + 5) * step_ms
    chunks: list[pd.DataFrame] = []
    remaining = bars + 5

    while remaining > 0:
        limit = min(1000, remaining)
        rows = api_get(
            f"{SPOT_DATA}/api/v3/klines",
            {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit},
            timeout=15,
        )
        if not rows:
            break
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["open_time", "close_time"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        chunks.append(df)
        last_open = int(df["open_time"].iloc[-1])
        new_start = last_open + step_ms
        if new_start <= start_ms:
            break
        start_ms = new_start
        remaining -= len(df)
        if len(df) < limit:
            break

    if not chunks:
        raise BinanceDataError("No historical Spot candles returned")

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    # Exclude the current unfinished candle from research outcomes.
    out = out[pd.to_numeric(out["close_time"], errors="coerce") < now_ms].tail(bars).copy()
    out["open_time"] = pd.to_datetime(out["open_time"].astype("int64"), unit="ms", utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"].astype("int64"), unit="ms", utc=True)
    return out.reset_index(drop=True)


def _series_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def historical_feature_frame(k: pd.DataFrame) -> pd.DataFrame:
    """Build only features that can be reconstructed from historical Spot klines."""
    x = k.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["rsi14"] = _series_rsi(x["close"], 14)
    x["ema_spread_pct"] = (x["ema20"] / x["ema50"] - 1) * 100
    x["price_vs_ema50_pct"] = (x["close"] / x["ema50"] - 1) * 100

    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    x["atr_pct"] = x["atr14"] / x["close"] * 100

    x["taker_sell_volume"] = (x["volume"] - x["taker_buy_volume"]).clip(lower=0)
    buy6 = x["taker_buy_volume"].rolling(6, min_periods=3).sum()
    sell6 = x["taker_sell_volume"].rolling(6, min_periods=3).sum()
    x["taker_ratio_6bars"] = (buy6 / sell6.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    trend = (
        (x["ema_spread_pct"] / 0.8).clip(-1, 1) * 0.45
        + (x["price_vs_ema50_pct"] / 2.0).clip(-1, 1) * 0.35
        + ((x["rsi14"] - 50) / 25).clip(-1, 1) * 0.20
    ).clip(-1, 1)
    taker_component = (np.log(x["taker_ratio_6bars"].clip(lower=1e-6)) / 0.25).clip(-1, 1)

    # Historical Spot data does not contain the live order-book snapshot and this cloud cannot
    # query Futures analytics. To avoid fabricating history, those layers are neutralized.
    # This mirrors the v0.1.1 fallback architecture as closely as reproducible data allows.
    order_flow_reconstructable = (0.7 * taker_component).clip(-1, 1)
    composite = (0.35 * trend + 0.25 * order_flow_reconstructable).clip(-1, 1)

    x["trend_score"] = trend
    x["order_flow_score"] = order_flow_reconstructable
    x["hist_composite"] = composite
    x["hist_score_100"] = composite * 100
    return x


def horizon_options(interval: str) -> dict[str, int]:
    if interval == "1h":
        return {"+1h": 1, "+4h": 4, "+12h": 12, "+24h": 24}
    return {"+4h": 1, "+12h": 3, "+24h": 6, "+48h": 12}


def build_nonoverlap_trades(
    x: pd.DataFrame,
    threshold_100: float,
    horizon_bars: int,
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    """Non-overlapping event trades so the equity curve and drawdown are interpretable."""
    rows: list[dict[str, Any]] = []
    i = 60  # indicator warm-up
    last_entry = len(x) - horizon_bars - 1
    cost = round_trip_cost_bps / 10_000.0
    threshold = threshold_100 / 100.0

    while i <= last_entry:
        score = float(x["hist_composite"].iloc[i])
        direction = 1 if score >= threshold else -1 if score <= -threshold else 0
        if direction == 0:
            i += 1
            continue

        entry = float(x["close"].iloc[i])
        exit_price = float(x["close"].iloc[i + horizon_bars])
        gross = direction * (exit_price / entry - 1.0)
        net = gross - cost
        rows.append({
            "entry_time": x["close_time"].iloc[i],
            "exit_time": x["close_time"].iloc[i + horizon_bars],
            "side": "LONG" if direction > 0 else "SHORT",
            "score": score * 100,
            "entry": entry,
            "exit": exit_price,
            "gross_return": gross,
            "net_return": net,
        })
        i += max(1, horizon_bars)  # prevent overlapping holding periods

    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0, "win_rate": np.nan, "expectancy": np.nan, "median": np.nan,
            "profit_factor": np.nan, "total_return": np.nan, "max_drawdown": np.nan,
            "avg_win": np.nan, "avg_loss": np.nan,
        }
    r = trades["net_return"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        "trades": float(len(trades)),
        "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "median": float(r.median()),
        "profit_factor": float(pf),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
    }


def chronological_holdout(trades: pd.DataFrame, train_fraction: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    cut = max(1, min(len(trades) - 1, int(len(trades) * train_fraction))) if len(trades) > 1 else 1
    return trades.iloc[:cut].copy(), trades.iloc[cut:].copy()


def score_bucket_study(x: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    y = x.copy()
    y["forward_return"] = y["close"].shift(-horizon_bars) / y["close"] - 1
    y = y.dropna(subset=["hist_score_100", "forward_return"]).copy()
    bins = [-100, -35, -15, 15, 35, 100]
    labels = ["≤ -35", "-35 to -15", "-15 to +15", "+15 to +35", "≥ +35"]
    y["score_bucket"] = pd.cut(y["hist_score_100"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label, g in y.groupby("score_bucket", observed=False):
        if g.empty:
            continue
        score_mid = float(g["hist_score_100"].mean())
        direction = -1 if score_mid < 0 else 1
        aligned = direction * g["forward_return"]
        rows.append({
            "Score bucket": str(label),
            "N": int(len(g)),
            "Mean fwd return %": float(g["forward_return"].mean() * 100),
            "Direction-aligned win %": float((aligned > 0).mean() * 100),
        })
    return pd.DataFrame(rows)

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                price REAL NOT NULL,
                regime TEXT NOT NULL,
                decision TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                features_json TEXT NOT NULL,
                risk_json TEXT NOT NULL
            )
        """)


def save_snapshot(symbol: str, interval: str, price: float, regime: str, signal: dict, features: dict, risk: dict) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """INSERT INTO snapshots(created_at,symbol,interval,price,regime,decision,score,confidence,features_json,risk_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(), symbol, interval, price, regime,
                signal["decision"], signal["score_100"], signal["confidence"],
                json.dumps(features, ensure_ascii=False), json.dumps(risk, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def load_snapshots(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id,created_at,symbol,interval,price,regime,decision,score,confidence FROM snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


st.set_page_config(page_title="Trading Research Engine v0.2", page_icon="📊", layout="wide")
st.title("Trading Research Engine v0.2")
st.caption("Live research + historical validation terminal. Public Binance market data only. No order execution.")

with st.sidebar:
    st.header("Market")
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=1)
    interval = st.selectbox("Timeframe", ["1h", "4h"], index=1)
    refresh = st.button("Refresh live data", type="primary", use_container_width=True)
    st.divider()
    st.caption("Live scores are hypotheses. v0.2 adds a reproducible Spot-history backtest before any claim of edge.")


@st.cache_data(ttl=60, show_spinner=False)
def get_analysis(symbol: str, interval: str):
    snap = snapshot(symbol, interval)
    f = extract_features(snap, interval)
    regime = detect_regime(f)
    signal = score_signal(f)
    risk = risk_plan(f, str(signal["decision"]))
    return snap, f, regime, signal, risk


if refresh:
    st.cache_data.clear()

try:
    snap, f, regime, signal, risk = get_analysis(symbol, interval)
except BinanceDataError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.exception(exc)
    st.stop()

source = str(snap.get("source", "Unknown source"))
limitations = str(snap.get("limitations", ""))
if "fallback" in source.lower():
    st.warning(f"Data source: {source}. {limitations}")
else:
    st.success(f"Data source: {source}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"${f['price']:,.2f}", f"{f['change_24h_pct']:+.2f}% / 24h")
c2.metric("Regime", regime)
c3.metric("Decision", signal["decision"])
c4.metric("Score", f"{signal['score_100']:+.1f} / 100")
c5.metric("Confidence", f"{float(signal['confidence']) * 100:.0f}%")

st.subheader("Signal decomposition")
score_df = pd.DataFrame({
    "Layer": ["Trend", "Derivatives", "Order flow", "Positioning"],
    "Score": [signal["trend"], signal["derivatives"], signal["order_flow"], signal["positioning"]],
}).set_index("Layer")
st.bar_chart(score_df, horizontal=True)

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Price")
    chart = snap["klines"].tail(120).set_index("open_time")[["close"]]
    st.line_chart(chart)

with right:
    st.subheader("Core features")
    feature_rows = [
        ("EMA20", f["ema20"]),
        ("EMA50", f["ema50"]),
        ("RSI14", f["rsi14"]),
        ("ATR %", f["atr_pct"]),
        ("OI Δ 24h %", f["oi_change_24h_pct"] if not snap["oi"].empty else np.nan),
        ("Funding %", f["funding_last_pct"] if not snap["funding"].empty else np.nan),
        ("Taker ratio", f["taker_ratio_6bars"]),
        ("Global L/S", f["global_ls"] if not snap["global_ls"].empty else np.nan),
        ("Top traders L/S", f["top_ls"] if not snap["top_ls"].empty else np.nan),
        ("Book bid/ask", f["book_bid_ask_ratio"]),
    ]
    st.dataframe(pd.DataFrame(feature_rows, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

st.subheader("Risk / scenario map")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Expected daily move", f"±${risk['expected_daily_move_usd']:,.0f}", f"{risk['expected_daily_move_pct']:.2f}%")
r2.metric("Reference entry", f"${risk['reference_entry']:,.2f}")
r3.metric("Invalidation", f"${risk['invalidation']:,.2f}")
r4.metric("Target 1", f"${risk['target1']:,.2f}", f"R:R {risk['rr1']:.2f}")
r5.metric("Target 2", f"${risk['target2']:,.2f}", f"R:R {risk['rr2']:.2f}")
st.caption(str(risk["note"]))


st.divider()
st.header("Historical validation — v0.2")
st.info(
    "This backtest uses closed Binance Spot candles only. It reconstructs Trend + historical taker-flow. "
    "Historical order-book imbalance, OI, funding and long/short positioning are unavailable from this reproducible feed, "
    "so this tests a simplified Spot score — not the exact live score above."
)

bt1, bt2, bt3, bt4 = st.columns([1.2, 1, 1, 1])
with bt1:
    bt_bars = st.selectbox("History", [1000, 2000, 3000], index=1, format_func=lambda n: f"{n:,} candles")
with bt2:
    bt_threshold = st.slider("Signal threshold", 15, 45, 35, 5, help="Absolute historical score required to create LONG/SHORT event trades.")
with bt3:
    horizon_map = horizon_options(interval)
    bt_horizon_label = st.selectbox("Exit horizon", list(horizon_map.keys()), index=min(2, len(horizon_map) - 1))
with bt4:
    bt_cost_bps = st.slider("Round-trip cost", 0, 30, 10, 1, help="Research assumption in basis points; user-adjustable, not a claim about your actual fee tier.")

run_bt = st.button("Run historical validation", type="primary", use_container_width=True)

@st.cache_data(ttl=1800, show_spinner=False)
def get_backtest_history(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    return historical_klines(symbol, interval, bars)

if run_bt:
    try:
        with st.spinner("Fetching closed Spot candles and testing non-overlapping events..."):
            hist = get_backtest_history(symbol, interval, int(bt_bars))
            hf = historical_feature_frame(hist)
            hb = int(horizon_map[bt_horizon_label])
            trades = build_nonoverlap_trades(hf, float(bt_threshold), hb, float(bt_cost_bps))
            train_trades, test_trades = chronological_holdout(trades, 0.70)
            all_stats = summarize_trades(trades)
            test_stats = summarize_trades(test_trades)
            buckets = score_bucket_study(hf.iloc[60:].copy(), hb)

        st.caption(
            f"Sample: {len(hist):,} closed {interval} candles · rule: |score| ≥ {bt_threshold} · "
            f"exit {bt_horizon_label} · non-overlapping trades · assumed round-trip cost {bt_cost_bps} bps."
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Trades", f"{int(all_stats['trades'])}")
        m2.metric("Win rate", "—" if np.isnan(all_stats['win_rate']) else f"{all_stats['win_rate']*100:.1f}%")
        m3.metric("Expectancy / trade", "—" if np.isnan(all_stats['expectancy']) else f"{all_stats['expectancy']*100:+.3f}%")
        pf = all_stats["profit_factor"]
        m4.metric("Profit factor", "—" if np.isnan(pf) else ("∞" if np.isinf(pf) else f"{pf:.2f}"))
        m5.metric("Compounded return", "—" if np.isnan(all_stats['total_return']) else f"{all_stats['total_return']*100:+.1f}%")
        m6.metric("Max drawdown", "—" if np.isnan(all_stats['max_drawdown']) else f"{all_stats['max_drawdown']*100:.1f}%")

        st.subheader("Chronological holdout: last 30% of trades")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Holdout trades", f"{int(test_stats['trades'])}")
        h2.metric("Holdout win rate", "—" if np.isnan(test_stats['win_rate']) else f"{test_stats['win_rate']*100:.1f}%")
        h3.metric("Holdout expectancy", "—" if np.isnan(test_stats['expectancy']) else f"{test_stats['expectancy']*100:+.3f}%")
        tpf = test_stats["profit_factor"]
        h4.metric("Holdout PF", "—" if np.isnan(tpf) else ("∞" if np.isinf(tpf) else f"{tpf:.2f}"))

        if int(all_stats["trades"]) < 30:
            st.warning("Too few non-overlapping trades for a stable conclusion. Increase history or lower the threshold; do not interpret this as evidence of edge.")
        elif (not np.isnan(test_stats["expectancy"])) and test_stats["expectancy"] > 0 and (np.isinf(tpf) or tpf > 1.0):
            st.success("The holdout is positive under these assumptions. This is a research lead, not proof: next step is walk-forward testing across more regimes and data sources.")
        else:
            st.warning("The holdout does not establish a positive edge under these assumptions. Treat the current score as unvalidated and revise/test rather than trade it.")

        st.subheader("Equity curve — non-overlapping research events")
        if not trades.empty:
            eq = (1 + trades["net_return"].astype(float)).cumprod()
            eq.index = pd.to_datetime(trades["exit_time"], utc=True)
            st.line_chart(pd.DataFrame({"equity": eq}))
            st.dataframe(
                trades.tail(25).assign(
                    score=lambda d: d["score"].round(1),
                    net_return_pct=lambda d: (d["net_return"] * 100).round(3),
                )[["entry_time", "exit_time", "side", "score", "entry", "exit", "net_return_pct"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No trades crossed the selected threshold in this history window.")

        st.subheader("Score-bucket event study")
        st.caption("Direction-aligned win % asks whether negative scores were followed by declines and positive scores by rises over the selected horizon.")
        st.dataframe(buckets, hide_index=True, use_container_width=True)

    except BinanceDataError as exc:
        st.error(f"Historical data request failed: {exc}")
    except Exception as exc:
        st.exception(exc)

st.subheader("Paper journal")
if st.button("Save this research snapshot"):
    row_id = save_snapshot(symbol, interval, f["price"], regime, signal, f, risk)
    st.success(f"Saved snapshot #{row_id}")

journal = load_snapshots(50)
if journal:
    st.dataframe(pd.DataFrame(journal), hide_index=True, use_container_width=True)
else:
    st.info("No saved snapshots yet.")

with st.expander("How v0.2 thinks"):
    st.markdown("""
- **Trend:** EMA20/EMA50, price vs EMA50, RSI.
- **Derivatives:** price/OI concordance and funding crowding penalty when Binance Futures is reachable; neutral in Spot fallback mode.
- **Order flow:** Futures taker ratio when available; otherwise Spot taker-buy volume proxy plus order-book imbalance.
- **Positioning:** top-trader/global long-short ratios when available; neutral in Spot fallback mode.
- **Decision:** LONG above +0.35, SHORT below -0.35, otherwise WAIT.

The live thresholds and weights are deliberately visible. The v0.2 historical section tests a reproducible Spot proxy and keeps unavailable historical layers neutral rather than fabricating them.
""")
