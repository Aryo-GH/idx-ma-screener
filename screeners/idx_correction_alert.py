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
def _compute_gate_values(df: pd.DataFrame, cfg: AlertConfig) -> dict:
    """Hitung semua nilai gate tanpa filter — untuk debug/diagnostik."""
    result = {"rows": len(df), "passed": False, "fail_gate": None}
    if len(df) < cfg.min_history:
        result["fail_gate"] = "min_history"
        return result

    df = compute_indicators(df)
    last = df.iloc[-1]

    close   = last["Close"]
    ma20    = last["MA20"]
    # Pakai nilai hari terakhir (bukan AvgValue20) — supaya saham yang baru
    # aktif setelah lama tidur (post-breakout) tidak terblokir liquidity gate
    last_vol   = last["Volume"] if not pd.isna(last["Volume"]) else 0
    last_value = close * last_vol if not pd.isna(close) else 0

    result.update({
        "close":     close,
        "ma20":      ma20,
        "avg_val_b": last_value / 1e9,
    })

    if pd.isna(close) or close < cfg.min_price:
        result["fail_gate"] = "min_price"; return result
    if last_value < cfg.min_avg_value_b * 1e9:
        result["fail_gate"] = "liquidity"; return result
    if any(pd.isna(last[c]) for c in ["MA5","MA20","AvgVol5","AvgVol20","StochK"]):
        result["fail_gate"] = "nan_indicator"; return result

    # Gate 1: prior peak
    lookback = min(cfg.lookback_peak, len(df) - 1)
    recent_slice = df.iloc[-(lookback + 1):-1]
    peak_close = recent_slice["Close"].max() if not recent_slice.empty else float("nan")
    peak_gap   = (peak_close / close - 1) * 100 if not pd.isna(peak_close) else float("nan")
    result["peak_close"] = peak_close
    result["peak_gap_pct"] = peak_gap
    if pd.isna(peak_close) or peak_close < close * (1 + cfg.min_peak_gap_pct):
        result["fail_gate"] = "prior_peak"; return result

    # Gate 2: drawdown
    drawdown = (peak_close - close) / peak_close * 100
    result["drawdown_pct"] = drawdown
    if drawdown > cfg.max_drawdown_pct * 100:
        result["fail_gate"] = "drawdown"; return result

    # Gate 3: above MA20
    pct_above_ma20 = (close / ma20 - 1) * 100 if not pd.isna(ma20) else float("nan")
    result["pct_above_ma20"] = pct_above_ma20
    if close < ma20 * cfg.min_above_ma20:
        result["fail_gate"] = "ma20_floor"; return result

    # Gate 4: volume
    vol_ratio = last["AvgVol5"] / last["AvgVol20"]
    result["vol_ratio_pct"] = vol_ratio * 100
    if pd.isna(vol_ratio) or vol_ratio >= cfg.vol_ratio_max:
        result["fail_gate"] = "volume"; return result

    # Gate 5: range 5d
    recent_5 = df.iloc[-5:]
    high_5d  = recent_5["High"].max() if "High" in recent_5.columns else recent_5["Close"].max()
    low_5d   = recent_5["Low"].min()  if "Low"  in recent_5.columns else recent_5["Close"].min()
    range_5d = (high_5d - low_5d) / close * 100
    result["range_5d_pct"] = range_5d
    if range_5d >= cfg.range_5d_max * 100:
        result["fail_gate"] = "range_5d"; return result

    # Gate 6: StochK
    stoch_k = last["StochK"]
    result["stoch_k"] = stoch_k
    if pd.isna(stoch_k) or stoch_k >= cfg.stoch_k_max:
        result["fail_gate"] = "stoch_k"; return result

    # ── Lolos semua ─────────────────────────────────────────────────────────
    peak_date = recent_slice["Close"].idxmax()
    days_since_peak = len(df) - 1 - df.index.get_loc(peak_date)
    result.update({
        "passed":         True,
        "fail_gate":      None,
        "days_since_peak": days_since_peak,
        "avg_value_b":    avg_val / 1e9,
    })
    return result


def check_alert(df: pd.DataFrame, cfg: AlertConfig) -> dict | None:
    """Return dict dengan detail kalau saham lolos, None kalau tidak lolos."""
    g = _compute_gate_values(df, cfg)
    if not g.get("passed"):
        return None

    # Rebuild full result dict dari g
    df = compute_indicators(df)
    last = df.iloc[-1]
    lookback = min(cfg.lookback_peak, len(df) - 1)
    recent_slice = df.iloc[-(lookback + 1):-1]
    peak_date  = recent_slice["Close"].idxmax()
    last_value = last["Close"] * last["Volume"] if not pd.isna(last["Volume"]) else 0

    return {
        "close":           g["close"],
        "ma20":            g["ma20"],
        "pct_above_ma20":  g["pct_above_ma20"],
        "peak_close":      g["peak_close"],
        "drawdown_pct":    g["drawdown_pct"],
        "days_since_peak": g["days_since_peak"],
        "vol_ratio_pct":   g["vol_ratio_pct"],
        "range_5d_pct":    g["range_5d_pct"],
        "stoch_k":         g["stoch_k"],
        "avg_value_b":     last_value / 1e9,
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
        # Pakai start/end explicit supaya yfinance selalu ambil data hari ini
        import datetime as _dt
        _today = _dt.date.today()
        _end   = (_today + _dt.timedelta(days=1)).isoformat()
        _period_days = {"1mo": 35, "2mo": 65, "3mo": 100, "6mo": 190,
                        "1y": 370, "60d": 65, "90d": 95}
        _start = (_today - _dt.timedelta(days=_period_days.get(period, 100))).isoformat()
        try:
            raw = yf.download(
                tickers=chunk, start=_start, end=_end, interval="1d",
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
# Debug / Funnel Analysis
# ---------------------------------------------------------------------------
def screen_debug(tickers: list[str], cfg: AlertConfig, top_n: int = 20) -> None:
    """
    Mode diagnostik: tampilkan berapa ticker gugur di setiap gate,
    dan tampilkan kandidat terdekat (hampir lolos) untuk tiap gate.
    """
    symbols = [to_yahoo_symbol(t) for t in tickers]
    print(f"\nDownload {len(symbols)} ticker dari yfinance...")
    data_map = _batch_download(symbols, period="3mo")
    print(f"Download selesai: {len(data_map)} ticker dengan data\n")

    GATES = ["min_history", "min_price", "liquidity", "nan_indicator",
             "prior_peak", "drawdown", "ma20_floor", "volume", "range_5d", "stoch_k"]
    gate_count  = {g: 0 for g in GATES}
    gate_samples = {g: [] for g in GATES}  # simpan (ticker, nilai relevan)

    passed = []
    no_data = 0

    for ticker, symbol in zip(tickers, symbols):
        raw_df = data_map.get(symbol.upper())
        if raw_df is None or raw_df.empty:
            no_data += 1
            continue
        try:
            g = _compute_gate_values(raw_df, cfg)
        except Exception as exc:
            gate_count.setdefault("error", 0)
            gate_count["error"] = gate_count.get("error", 0) + 1
            continue

        if g.get("passed"):
            passed.append(ticker.upper())
        else:
            fg = g.get("fail_gate", "unknown")
            gate_count[fg] = gate_count.get(fg, 0) + 1
            gate_samples[fg].append((ticker.upper(), g))

    # ── Tampilkan funnel ─────────────────────────────────────────────────────
    total = len(tickers)
    print("=" * 60)
    print(f"FUNNEL ANALYSIS  ({total} ticker diproses, {no_data} tanpa data)")
    print("=" * 60)
    gate_labels = {
        "min_history":   "Riwayat <40 hari",
        "min_price":     "Harga <50 IDR",
        "liquidity":     "Likuiditas <1B IDR/hari",
        "nan_indicator": "Indikator NaN",
        "prior_peak":    "Tidak ada prior peak ≥10%",
        "drawdown":      "Drawdown >45% (terlalu crash)",
        "ma20_floor":    "Harga <MA20×0.88",
        "volume":        "Volume tidak turun (<65%)",
        "range_5d":      "Range 5 hari terlalu lebar (>8%)",
        "stoch_k":       "StochK tidak oversold (>30)",
        "error":         "Error saat kalkulasi",
    }
    for gate, label in gate_labels.items():
        cnt = gate_count.get(gate, 0)
        if cnt > 0:
            bar = "█" * min(cnt // max(total // 40, 1), 40)
            print(f"  {label:<38} {cnt:>4}  {bar}")

    print(f"\n  {'LOLOS SEMUA GATE':<38} {len(passed):>4}  {'★' * len(passed)}")
    if passed:
        print(f"  → {', '.join(passed)}")

    # ── Kandidat terdekat per gate ───────────────────────────────────────────
    # Untuk volume & range & stochK: tampilkan yang paling dekat threshold
    def show_near_miss(gate: str, sort_key, reverse: bool, fmt: str, label: str, threshold: str):
        samples = gate_samples.get(gate, [])
        valid = [(t, g) for t, g in samples if not pd.isna(g.get(sort_key, float("nan")))]
        if not valid:
            return
        valid.sort(key=lambda x: x[1].get(sort_key, 0), reverse=reverse)
        print(f"\n  {label} (threshold: {threshold})  — {len(samples)} gagal, terdekat:")
        for t, g in valid[:top_n]:
            val = g.get(sort_key, float("nan"))
            print(f"    {t:<8} {fmt.format(val)}")

    print("\n" + "─" * 60)
    print("NEAR-MISS DETAIL (kandidat paling dekat lolos)")
    print("─" * 60)

    show_near_miss("volume",   "vol_ratio_pct", False, "{:.0f}%",   "VOLUME ratio (mau <65%)",    "<65%")
    show_near_miss("range_5d", "range_5d_pct",  False, "{:.1f}%",   "RANGE 5 hari (mau <8%)",     "<8%")
    show_near_miss("stoch_k",  "stoch_k",       False, "{:.1f}",    "STOCH RSI K (mau <30)",      "<30")
    show_near_miss("prior_peak","peak_gap_pct", True,  "{:.1f}%",   "PRIOR PEAK gap (mau ≥10%)",  "≥10%")
    show_near_miss("ma20_floor","pct_above_ma20",True, "{:+.1f}%",  "DI BAWAH MA20 (mau >-12%)",  ">-12%")
    show_near_miss("drawdown", "drawdown_pct",  False, "{:.1f}%",   "DRAWDOWN (mau <45%)",        "<45%")

    print()


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

    # Gunakan tanggal data aktual (candle terakhir) jika ada, fallback ke hari ini
    if hits:
        data_date = hits[0].get("as_of", date.today().isoformat())
        try:
            from datetime import datetime
            d = datetime.strptime(data_date, "%Y-%m-%d")
            date_str = d.strftime("%-d %b %Y")
        except Exception:
            date_str = data_date
    else:
        date_str = date.today().strftime("%-d %b %Y")

    lines = [
        f"📉 *CORRECTION ALERT* — {date_str}",
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
    parser.add_argument("--debug",        action="store_true",
                        help="Mode diagnostik: tampilkan funnel berapa ticker gugur di setiap gate")
    parser.add_argument("--top-n",        type=int,   default=10,
                        help="Jumlah near-miss yang ditampilkan per gate di --debug (default 10)")
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

    # Mode debug: tampilkan funnel, lalu keluar
    if args.debug:
        screen_debug(tickers, cfg, top_n=args.top_n)
        return

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
