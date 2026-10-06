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
        "note": "Reference levels only; v0.1 has not yet demonstrated out-of-sample edge.",
    }


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


st.set_page_config(page_title="Trading Research Engine v0.1.1", page_icon="📊", layout="wide")
st.title("Trading Research Engine v0.1.1")
st.caption("Research + paper-trading terminal. Public Binance market data only. No order execution.")

with st.sidebar:
    st.header("Market")
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=1)
    interval = st.selectbox("Timeframe", ["1h", "4h"], index=1)
    refresh = st.button("Refresh live data", type="primary", use_container_width=True)
    st.divider()
    st.caption("v0.1 scores are research hypotheses. They must be backtested before being treated as an edge.")


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

st.subheader("Paper journal")
if st.button("Save this research snapshot"):
    row_id = save_snapshot(symbol, interval, f["price"], regime, signal, f, risk)
    st.success(f"Saved snapshot #{row_id}")

journal = load_snapshots(50)
if journal:
    st.dataframe(pd.DataFrame(journal), hide_index=True, use_container_width=True)
else:
    st.info("No saved snapshots yet.")

with st.expander("How v0.1 thinks"):
    st.markdown("""
- **Trend:** EMA20/EMA50, price vs EMA50, RSI.
- **Derivatives:** price/OI concordance and funding crowding penalty when Binance Futures is reachable; neutral in Spot fallback mode.
- **Order flow:** Futures taker ratio when available; otherwise Spot taker-buy volume proxy plus order-book imbalance.
- **Positioning:** top-trader/global long-short ratios when available; neutral in Spot fallback mode.
- **Decision:** LONG above +0.35, SHORT below -0.35, otherwise WAIT.

The thresholds and weights are deliberately visible. They are **research hypotheses**, not validated alpha.
""")
