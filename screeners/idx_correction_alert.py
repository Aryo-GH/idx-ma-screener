#!/usr/bin/env python3
"""
idx_correction_alert.py

Alert Telegram untuk pola "Correction Low Volume — Sideways":
  Saham yang sedang dalam fase koreksi SEHAT setelah prior upswing:
    1. Ada prior move up  (peak 20-30 hari lalu ≥ 10% di atas harga sekarang)
    2. Koreksi orderly   (range 5 hari < 8%, tidak crash)
    3. Volume mengecil   (AvgVol5 / AvgVol20 < 65%)
    4. Stoch RSI oversold (StochK < 30)
    5. Harga belum break   (close > MA20 × 0.88)
    6. Likuiditas cukup    (avg value > 1 miliar IDR/hari)

Setup ini sering muncul sebelum bounce / re-test MA atau breakout lanjutan.

CONTOH PAKAI
------------
    # Test beberapa ticker
    python idx_correction_alert.py --tickers BBCA,IATA,NASI,BREN

    # Screen seluruh universe
    python idx_correction_alert.py --tickers-file data/idx_universe.txt --telegram

    # Via GitHub Actions (env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    python idx_correction_alert.py --tickers-file data/idx_universe.txt --telegram
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Konfigurasi threshold (tuning di sini)
# ---------------------------------------------------------------------------
@dataclass
class AlertConfig:
    # Prior upswing: cek max harga dalam N hari terakhir
    lookback_peak: int = 30         # cari peak dalam 30 hari ke belakang
    min_peak_gap_pct: float = 0.10  # peak harus ≥10% di atas harga sekarang

    # Koreksi masih orderly (tidak crash)
    max_drawdown_pct: float = 0.45  # max koreksi dari peak (lebih dari 45% = terlalu dalam)
    min_above_ma20: float = 0.88    # close > MA20 × 0.88 (masih dekat MA20)

    # Volume mengecil
    vol_ratio_max: float = 0.65     # AvgVol5 / AvgVol20 < 65%

    # Sideways (range 5 hari sempit)
    range_5d_max: float = 0.08      # (high5d - low5d) / close < 8%

    # Stoch RSI oversold
    stoch_k_max: float = 30.0       # StochK < 30

    # Minimum history
    min_history: int = 40

    # Liquidity
    min_avg_value_b: float = 1.0    # avg value ≥ 1 miliar IDR/hari
    min_price: float = 50.0         # min last close


# ---------------------------------------------------------------------------
# Indikator teknikal
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(series: pd.Series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi_vals = _rsi(series, rsi_period)
    min_rsi = rsi_vals.rolling(stoch_period).min()
    max_rsi = rsi_vals.rolling(stoch_period).max()
    stoch = (rsi_vals - min_rsi) / (max_rsi - min_rsi).replace(0, pd.NA) * 100
    k = stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"]      = df["Close"].rolling(5).mean()
    df["MA20"]     = df["Close"].rolling(20).mean()
    df["AvgVol5"]  = df["Volume"].rolling(5).mean()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    if "Value" not in df.columns:
        df["Value"] = df["Close"] * df["Volume"]
    df["AvgValue20"] = df["Value"].rolling(20).mean()
    df["StochK"], df["StochD"] = _stoch_rsi(df["Close"])
    return df


# ---------------------------------------------------------------------------
# Core check: apakah saham ini masuk kriteria?
# ---------------------------------------------------------------------------
def check_alert(df: pd.DataFrame, cfg: AlertConfig) -> dict | None:
    """Return dict dengan detail kalau saham lolos, None kalau tidak lolos."""
    if len(df) < cfg.min_history:
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    # ── Liquidity gate ──────────────────────────────────────────────────────
    close = last["Close"]
    if pd.isna(close) or close < cfg.min_price:
        return None

    avg_val = last["AvgValue20"]
    if pd.isna(avg_val) or avg_val < cfg.min_avg_value_b * 1e9:
        return None

    # ── Gate: indikator harus valid ─────────────────────────────────────────
    for col in ["MA5", "MA20", "AvgVol5", "AvgVol20", "StochK"]:
        if pd.isna(last[col]):
            return None

    ma20 = last["MA20"]

    # ── Gate 1: Prior upswing — cari peak dalam lookback_peak hari terakhir ─
    lookback = min(cfg.lookback_peak, len(df) - 1)
    recent_slice = df.iloc[-(lookback + 1):-1]  # tidak termasuk candle hari ini
    if recent_slice.empty:
        return None

    peak_close = recent_slice["Close"].max()
    peak_date  = recent_slice["Close"].idxmax()

    # Peak harus ≥ min_peak_gap_pct di atas harga sekarang
    if pd.isna(peak_close) or peak_close < close * (1 + cfg.min_peak_gap_pct):
        return None

    # ── Gate 2: Koreksi tidak terlalu dalam ─────────────────────────────────
    drawdown = (peak_close - close) / peak_close
    if drawdown > cfg.max_drawdown_pct:
        return None  # sudah terlalu crash, bukan koreksi sehat

    # ── Gate 3: Harga masih di atas MA20 × threshold ────────────────────────
    if close < ma20 * cfg.min_above_ma20:
        return None

    # ── Gate 4: Volume mengecil ──────────────────────────────────────────────
    vol_ratio = last["AvgVol5"] / last["AvgVol20"]
    if pd.isna(vol_ratio) or vol_ratio >= cfg.vol_ratio_max:
        return None

    # ── Gate 5: Sideways — range 5 hari sempit ──────────────────────────────
    recent_5 = df.iloc[-5:]
    high_5d  = recent_5["High"].max() if "High" in recent_5.columns else recent_5["Close"].max()
    low_5d   = recent_5["Low"].min()  if "Low"  in recent_5.columns else recent_5["Close"].min()
    range_5d = (high_5d - low_5d) / close
    if range_5d >= cfg.range_5d_max:
        return None

    # ── Gate 6: Stoch RSI oversold ───────────────────────────────────────────
    stoch_k = last["StochK"]
    if pd.isna(stoch_k) or stoch_k >= cfg.stoch_k_max:
        return None

    # ── Lolos semua gate — kumpulkan detail ─────────────────────────────────
    days_since_peak = len(df) - 1 - df.index.get_loc(peak_date)

    return {
        "close":            close,
        "ma20":             ma20,
        "pct_above_ma20":   (close / ma20 - 1) * 100,
        "peak_close":       peak_close,
        "drawdown_pct":     drawdown * 100,
        "days_since_peak":  days_since_peak,
        "vol_ratio_pct":    vol_ratio * 100,
        "range_5d_pct":     range_5d * 100,
        "stoch_k":          stoch_k,
        "avg_value_b":      avg_val / 1e9,
    }


# ---------------------------------------------------------------------------
# Data loading (yfinance, fallback, no Arjum needed — simple alert)
# ---------------------------------------------------------------------------
def to_yahoo_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".JK"):
        t = f"{t}.JK"
    return t


def _batch_download(symbols: list[str], period: str = "3mo",
                    chunk_size: int = 200) -> dict[str, pd.DataFrame]:
    """Chunked batch download dari yfinance."""
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    print(f"  → {len(symbols)} ticker, {len(chunks)} chunk...")

    for idx, chunk in enumerate(chunks):
        print(f"  → Chunk {idx + 1}/{len(chunks)} ({len(chunk)} ticker)...")
        try:
            raw = yf.download(
                tickers=chunk, period=period, interval="1d",
                auto_adjust=False, group_by="ticker",
                threads=True, progress=False,
            )
        except Exception as exc:
            print(f"  [warn] Chunk {idx + 1} gagal: {exc}")
            continue
        if raw is None or raw.empty:
            continue

        if len(chunk) == 1:
            sym = chunk[0].upper()
            df = raw.copy()
            df.columns = [c.title() if isinstance(c, str) else c[0].title() for c in df.columns]
            req = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df.columns for c in req):
                df = df[req].astype(float).dropna(how="all")
                if not df.empty:
                    result[sym] = df
            continue

        for sym in chunk:
            try:
                df = raw[sym].copy()
                df.columns = [c.title() if isinstance(c, str) else str(c).title() for c in df.columns]
                req = ["Open", "High", "Low", "Close", "Volume"]
                if not all(c in df.columns for c in req):
                    continue
                df = df[req].astype(float).dropna(how="all")
                if not df.empty:
                    result[sym.upper()] = df
            except (KeyError, AttributeError):
                pass

    return result


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(token: str, chat_id: str, text: str) -> bool:
    import json
    import urllib.request

    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text,
                       "parse_mode": "Markdown"}).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15):
            return True
    except Exception as exc:
        print(f"[telegram error] {exc}")
        return False


# ---------------------------------------------------------------------------
# Main screening
# ---------------------------------------------------------------------------
def screen(tickers: list[str], cfg: AlertConfig) -> list[dict]:
    symbols = [to_yahoo_symbol(t) for t in tickers]
    print(f"Download {len(symbols)} ticker dari yfinance...")
    data_map = _batch_download(symbols, period="3mo")
    print(f"Download selesai: {len(data_map)} ticker")

    hits = []
    for ticker, symbol in zip(tickers, symbols):
        raw_df = data_map.get(symbol.upper())
        if raw_df is None or raw_df.empty:
            continue
        try:
            result = check_alert(raw_df, cfg)
            if result is not None:
                result["ticker"] = ticker.upper()
                result["as_of"]  = raw_df.index[-1].date().isoformat()
                hits.append(result)
        except Exception as exc:
            print(f"[error] {ticker}: {exc}")

    # Sort by Stoch RSI (terendah = paling oversold = paling menarik)
    hits.sort(key=lambda x: x["stoch_k"])
    return hits


def format_telegram_msg(hits: list[dict], total: int) -> str:
    from datetime import date

    today = date.today().strftime("%-d %b %Y")
    lines = [
        f"📉 *CORRECTION ALERT* — {today}",
        f"Koreksi orderly · vol turun · Stoch RSI oversold",
        f"*{len(hits)}* kandidat dari {total} saham\n",
    ]

    if hits:
        # Header tabel
        hdr = f"{'Ticker':<6} {'Harga':>6} {'K':>5} {'Vol':>4} {'Rng':>5} {'MA20':>6}"
        sep = "─" * len(hdr)
        rows = [hdr, sep]
        for h in hits:
            rows.append(
                f"{h['ticker']:<6} {h['close']:>6.0f} "
                f"{h['stoch_k']:>4.1f} "
                f"{h['vol_ratio_pct']:>3.0f}% "
                f"{h['range_5d_pct']:>4.1f}% "
                f"{h['pct_above_ma20']:>+5.1f}%"
            )
        # Kirim sebagai code block agar monospace & rapi
        lines.append("```")
        lines.extend(rows)
        lines.append("```")
        lines.append("_K=StochRSI · Vol=avg5/20 · Rng=range5d · MA20=jarak_")
    else:
        lines.append("_Tidak ada kandidat hari ini._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Alert Telegram: koreksi + low volume + Stoch RSI oversold (IDX)."
    )
    parser.add_argument("--tickers",      default=None,
                        help="Comma-separated ticker (mis. BBCA,TOWR,IATA)")
    parser.add_argument("--tickers-file", default=None,
                        help="File satu ticker per baris")
    parser.add_argument("--telegram",     action="store_true",
                        help="Kirim hasil ke Telegram (perlu env TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    parser.add_argument("--min-value-b",  type=float, default=1.0,
                        help="Min avg value per hari (miliar IDR), default 1.0")
    parser.add_argument("--vol-ratio",    type=float, default=0.65,
                        help="Maks AvgVol5/AvgVol20 (default 0.65 = 65%%)")
    parser.add_argument("--stoch-max",    type=float, default=30.0,
                        help="Maks Stoch RSI K (default 30)")
    parser.add_argument("--range-max",    type=float, default=0.08,
                        help="Maks range 5 hari / close (default 0.08 = 8%%)")
    args = parser.parse_args()

    # Kumpulkan ticker
    tickers: list[str] = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        p = Path(args.tickers_file)
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip().upper()
                if ln and not ln.startswith("#"):
                    tickers.append(ln)
    tickers = list(dict.fromkeys(tickers))  # deduplicate

    if not tickers:
        print("Tidak ada ticker. Pakai --tickers atau --tickers-file.")
        return

    cfg = AlertConfig(
        min_avg_value_b=args.min_value_b,
        vol_ratio_max=args.vol_ratio,
        stoch_k_max=args.stoch_max,
        range_5d_max=args.range_max,
    )

    print(f"\n=== Correction Low Volume Alert ===")
    print(f"Threshold: vol_ratio<{cfg.vol_ratio_max:.0%}, StochK<{cfg.stoch_k_max}, range<{cfg.range_5d_max:.0%}")
    print(f"Ticker: {len(tickers)}\n")

    hits = screen(tickers, cfg)

    # Tampilkan di terminal
    print(f"\n{'─'*60}")
    print(f"HASIL: {len(hits)} kandidat dari {len(tickers)} saham")
    print(f"{'─'*60}")
    for h in hits:
        print(
            f"{h['ticker']:6s}  Rp{h['close']:.0f}  "
            f"drawdown {h['drawdown_pct']:.1f}%  "
            f"vol {h['vol_ratio_pct']:.0f}%  "
            f"StochK {h['stoch_k']:.1f}  "
            f"range {h['range_5d_pct']:.1f}%"
        )

    # Kirim Telegram
    if args.telegram:
        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            print("[warn] TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID tidak di-set, skip Telegram.")
        else:
            msg = format_telegram_msg(hits, len(tickers))
            ok  = send_telegram(token, chat_id, msg)
            print(f"\n[telegram] {'Terkirim ✓' if ok else 'Gagal ✗'}")


if __name__ == "__main__":
    main()
