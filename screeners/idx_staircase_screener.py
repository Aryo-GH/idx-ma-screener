#!/usr/bin/env python3
"""
idx_staircase_screener.py

Screener untuk pola "tangga" (staircase) IDX.
Mencari saham yang naik seperti anak tangga:
  konsolidasi sideways (landing) → kenaikan tajam (step) → landing → step → ...

Contoh ideal: NASI (Jul–Sep 2026): dari ~100 ke 280 dalam 3 bulan
dengan 3–4 siklus tangga dan akumulasi bandar yang clean.

─────────────────────────────────────────────────────────────
ALGORITMA DETEKSI SIKLUS
─────────────────────────────────────────────────────────────
Landing (konsolidasi):
  - ≥3 hari berturut-turut
  - (High-Low)/Close < 2.5% per hari (candle ketat)
  - |daily return| < 2% per hari

Step (kenaikan):
  - Harga naik ≥4% total dari akhir landing
  - Terjadi dalam ≤7 hari setelah landing berakhir

Satu siklus = satu landing + satu step yang mengikutinya.
Dibutuhkan minimal 2 siklus dalam 40 hari terakhir.

─────────────────────────────────────────────────────────────
SISTEM SKORING 0–100
─────────────────────────────────────────────────────────────
  A. Jumlah siklus landing→step (30 pts)
  B. Kualitas konsolidasi – seberapa flat landing (25 pts)
  C. Kualitas kenaikan – gain + volume confirmation (25 pts)
  D. Trend keseluruhan – MA5>MA20, arah MAs (20 pts)

─────────────────────────────────────────────────────────────
PEMAKAIAN
─────────────────────────────────────────────────────────────
  # Test beberapa ticker langsung
  python idx_staircase_screener.py --tickers NASI,BBCA,TLKM,MUTU

  # Screen full universe
  python idx_staircase_screener.py \\
      --tickers-file data/idx_universe.txt \\
      --min-score 60 --top 20

  # Jalankan dan kirim ke Telegram (butuh env TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
  python idx_staircase_screener.py \\
      --tickers-file data/idx_universe.txt \\
      --min-score 60 --telegram
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Konfigurasi & threshold
# ---------------------------------------------------------------------------
@dataclass
class StaircaseConfig:
    # ─── Window deteksi pola ────────────────────────────────────
    lookback_days: int = 40           # lihat 40 hari terakhir (~8 minggu)

    # ─── Definisi "landing" (konsolidasi) ───────────────────────
    landing_min_days: int = 3         # ≥3 hari berturut-turut quiet
    landing_max_range_pct: float = 0.025   # (High-Low)/Close < 2.5%
    landing_max_daily_ret: float = 0.020   # |return harian| < 2%

    # ─── Definisi "step" (kenaikan) ─────────────────────────────
    step_min_gain_pct: float = 0.040  # total gain ≥4%
    step_max_days: int = 7            # step selesai dalam ≤7 hari

    # ─── Syarat minimum pola ────────────────────────────────────
    min_cycles: int = 2               # ≥2 siklus (landing→step)
    min_pattern_span: int = 10        # pola harus span ≥10 hari trading

    # ─── Filter likuiditas & harga ──────────────────────────────
    min_avg_value_b: float = 1.0      # avg value ≥1 miliar IDR/hari
    min_price: float = 50.0           # close ≥50 IDR

    # ─── Data minimum ────────────────────────────────────────────
    min_history: int = 70             # ≥70 candle untuk hitung MA20 + pattern

    # ─── Bobot skor ─────────────────────────────────────────────
    w_cycles: int = 30
    w_landing: int = 25
    w_step: int = 25
    w_trend: int = 20


DEFAULT_CFG = StaircaseConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_yahoo_symbol(ticker: str) -> str:
    t = ticker.upper().strip()
    return t if t.endswith(".JK") else f"{t}.JK"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan MA5, MA20, slope, dan value ke DataFrame."""
    df = df.copy()
    df["MA5"]       = df["Close"].rolling(5).mean()
    df["MA20"]      = df["Close"].rolling(20).mean()
    df["MA5_slope"] = (df["MA5"]  - df["MA5"].shift(5))  / df["MA5"].shift(5)
    df["MA20_slope"]= (df["MA20"] - df["MA20"].shift(5)) / df["MA20"].shift(5)
    df["Value"]     = df["Close"] * df["Volume"]          # IDR (Close × lembar)
    df["AvgVal20"]  = df["Value"].rolling(20).mean()
    return df


# ---------------------------------------------------------------------------
# Core: deteksi siklus landing → step
# ---------------------------------------------------------------------------
def find_cycles(df_recent: pd.DataFrame, cfg: StaircaseConfig) -> list[dict]:
    """
    Temukan siklus tangga dalam price DataFrame (subset 'lookback_days' candle).

    Algoritma:
      1. Klasifikasikan tiap hari: quiet jika range<2.5% DAN |ret|<2%
      2. Cari deretan ≥3 hari quiet → landing
      3. Setelah landing, cari step: gain ≥4% dalam ≤7 hari
      4. Kalau ketemu → satu siklus. Lanjut dari hari setelah step.

    Returns list of cycle dicts.
    """
    n = len(df_recent)
    if n < cfg.min_pattern_span:
        return []

    highs  = df_recent["High"].values
    lows   = df_recent["Low"].values
    closes = df_recent["Close"].values
    vols   = df_recent["Volume"].values

    # Metrik harian
    range_pct = (highs - lows) / closes
    rets = pd.Series(closes).pct_change().fillna(0).values

    # Tag: True = hari quiet (kandidat bagian landing)
    is_quiet = (range_pct < cfg.landing_max_range_pct) & (abs(rets) < cfg.landing_max_daily_ret)

    cycles: list[dict] = []
    i = 0

    while i < n:
        # Lewati hari non-quiet
        if not is_quiet[i]:
            i += 1
            continue

        # Hitung berapa hari quiet berturut-turut
        j = i
        while j < n and is_quiet[j]:
            j += 1
        landing_len = j - i

        if landing_len < cfg.landing_min_days:
            i = j
            continue

        # Landing valid: index i..j-1
        land_avg_range = float(range_pct[i:j].mean())
        land_vol       = float(vols[i:j].mean())

        # Cari step mulai dari j (boleh 1 hari gap, tapi tidak lebih)
        step_found = False
        for step_s in range(j, min(j + 2, n)):          # 0 atau 1 hari gap
            for step_e in range(step_s, min(step_s + cfg.step_max_days, n)):
                gain = closes[step_e] / closes[step_s] - 1
                if gain >= cfg.step_min_gain_pct:
                    step_vol  = float(vols[step_s: step_e + 1].mean())
                    vol_ratio = step_vol / land_vol if land_vol > 0 else 1.0

                    cycles.append(dict(
                        landing_start     = i,
                        landing_end       = j - 1,
                        landing_days      = landing_len,
                        landing_avg_range = land_avg_range,
                        step_start        = step_s,
                        step_end          = step_e,
                        step_days         = step_e - step_s + 1,
                        step_gain         = float(gain),
                        vol_ratio         = float(vol_ratio),
                    ))
                    i = step_e + 1
                    step_found = True
                    break
            if step_found:
                break

        if not step_found:
            i = j   # Landing ada tapi tidak diikuti step; lanjut

    return cycles


# ---------------------------------------------------------------------------
# Scoring satu saham
# ---------------------------------------------------------------------------
def score_stock(df: pd.DataFrame, cfg: StaircaseConfig) -> dict | None:
    """
    Score saham untuk pola tangga. Return None jika tidak lolos filter.

    Return dict berisi skor dan metrik pola kalau lolos.
    """
    if len(df) < cfg.min_history:
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    # ── Filter harga minimum ──────────────────────────────────────
    if pd.isna(last["Close"]) or last["Close"] < cfg.min_price:
        return None

    # ── Filter likuiditas ─────────────────────────────────────────
    avg_val_b = last["AvgVal20"] / 1e9 if pd.notna(last["AvgVal20"]) else 0.0
    if avg_val_b < cfg.min_avg_value_b:
        return None

    # ── Gate trend: harus dalam uptrend ──────────────────────────
    if pd.isna(last["MA5"]) or pd.isna(last["MA20"]):
        return None
    if last["MA5"] <= last["MA20"]:
        return None          # MA5 tidak di atas MA20
    if last["Close"] < last["MA20"]:
        return None          # Harga di bawah MA20 = setup rusak
    if pd.isna(last["MA5_slope"]) or last["MA5_slope"] <= 0:
        return None          # MA5 tidak naik
    if pd.isna(last["MA20_slope"]) or last["MA20_slope"] <= 0:
        return None          # MA20 tidak naik

    # ── Deteksi siklus tangga ─────────────────────────────────────
    df_recent = df.tail(cfg.lookback_days).copy().reset_index(drop=True)
    cycles = find_cycles(df_recent, cfg)

    if len(cycles) < cfg.min_cycles:
        return None

    # Span pola harus cukup panjang
    span = cycles[-1]["step_end"] - cycles[0]["landing_start"] + 1
    if span < cfg.min_pattern_span:
        return None

    # ── Rata-rata metrik siklus ───────────────────────────────────
    num_c          = len(cycles)
    avg_land_range = sum(c["landing_avg_range"] for c in cycles) / num_c
    avg_step_gain  = sum(c["step_gain"]         for c in cycles) / num_c
    avg_vol_ratio  = sum(c["vol_ratio"]          for c in cycles) / num_c

    # ── A: Jumlah siklus (30 pts) ─────────────────────────────────
    # 2 siklus → 18 | 3 → 26 | 4+ → 30
    score_a = min(30, 10 + (num_c - 2) * 8 + (2 if num_c == 2 else 0))
    # Lebih tepat:
    if num_c >= 4:   score_a = 30
    elif num_c == 3: score_a = 24
    else:            score_a = 16   # num_c == 2

    # ── B: Kualitas landing (25 pts) ─────────────────────────────
    # avg_land_range lebih kecil = landing lebih flat = skor lebih tinggi
    if avg_land_range < 0.010:   score_b = 25
    elif avg_land_range < 0.015: score_b = 20
    elif avg_land_range < 0.020: score_b = 13
    else:                        score_b = 7

    # ── C: Kualitas step (25 pts) ────────────────────────────────
    # Gain: 1 pt per % gain, max 15
    # Volume uplift: 10 kalau vol step ≥2× landing, 6 kalau ≥1.3×, 2 sisanya
    gain_pts = min(15, int(avg_step_gain * 100))
    vol_pts  = 10 if avg_vol_ratio >= 2.0 else (6 if avg_vol_ratio >= 1.3 else 2)
    score_c  = min(25, gain_pts + vol_pts)

    # ── D: Trend keseluruhan (20 pts) ────────────────────────────
    ma_ratio = last["MA5"] / last["MA20"]
    d = 0
    # MA5 naik setidaknya 1% dalam 5 hari
    d += 6 if last["MA5_slope"]  > 0.01 else 3
    # MA20 naik setidaknya 0.5% dalam 5 hari
    d += 6 if last["MA20_slope"] > 0.005 else 3
    # MA5 jelas di atas MA20
    d += 4 if ma_ratio >= 1.03 else (2 if ma_ratio >= 1.01 else 0)
    # Close masih dekat MA5 (nempel, belum terlalu jauh)
    close_vs_ma5 = last["Close"] / last["MA5"]
    d += 4 if 0.97 <= close_vs_ma5 <= 1.10 else 0
    score_d = min(20, d)

    total = score_a + score_b + score_c + score_d

    # Info siklus terakhir (untuk laporan)
    last_cyc = cycles[-1]
    last_step_gain = last_cyc["step_gain"]

    return {
        "score":               total,
        "score_raw":           total,
        "A":                   score_a,
        "B":                   score_b,
        "C":                   score_c,
        "D":                   score_d,
        "num_cycles":          num_c,
        "avg_step_gain_pct":   round(avg_step_gain * 100, 1),
        "last_step_gain_pct":  round(last_step_gain * 100, 1),
        "avg_vol_ratio":       round(avg_vol_ratio, 1),
        "avg_land_range_pct":  round(avg_land_range * 100, 2),
        "pattern_span_days":   span,
        "last_close":          round(float(last["Close"]), 0),
        "ma5":                 round(float(last["MA5"]), 0),
        "ma20":                round(float(last["MA20"]), 0),
        "avg_value_b":         round(avg_val_b, 1),
    }


# ---------------------------------------------------------------------------
# Batch download (reuse dari MA screener, chunked untuk avoid timeout)
# ---------------------------------------------------------------------------
def _batch_download(symbols: list[str], period: str, interval: str,
                    chunk_size: int = 200) -> dict[str, pd.DataFrame]:
    """Download ticker dalam chunk supaya tidak timeout / partial result."""
    import yfinance as yf

    result: dict[str, pd.DataFrame] = {}
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    print(f"  → {len(symbols)} ticker dibagi {len(chunks)} chunk (@{chunk_size})")

    for ci, chunk in enumerate(chunks):
        print(f"  → Chunk {ci + 1}/{len(chunks)} ({len(chunk)} ticker)…")
        # Pakai start/end explicit supaya yfinance selalu ambil data hari ini
        import datetime as _dt
        _today = _dt.date.today()
        _end   = (_today + _dt.timedelta(days=1)).isoformat()
        _period_days = {"1mo": 35, "2mo": 65, "3mo": 100, "6mo": 190,
                        "1y": 370, "2y": 740, "60d": 65, "90d": 95}
        _start = (_today - _dt.timedelta(days=_period_days.get(period, 100))).isoformat()
        try:
            raw = yf.download(
                tickers=chunk,
                start=_start,
                end=_end,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            print(f"  [warn] Chunk {ci + 1} gagal: {exc}")
            continue

        if raw is None or raw.empty:
            continue

        # Satu ticker → kolom flat
        if len(chunk) == 1:
            sym = chunk[0].upper()
            df = raw.copy()
            df.columns = [c.title() if isinstance(c, str) else c[0].title()
                          for c in df.columns]
            req = ["Open", "High", "Low", "Close", "Volume"]
            if all(c in df.columns for c in req):
                df = df[req].astype(float).dropna(how="all")
                if not df.empty:
                    result[sym] = df
            continue

        # Banyak ticker → MultiIndex (metric, ticker)
        for sym in chunk:
            try:
                df = raw[sym].copy()
                df.columns = [c.title() if isinstance(c, str) else str(c).title()
                              for c in df.columns]
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
# ---------------------------------------------------------------------------
# Arjum API (IDX Exchange data — aktif kalau env ARJUM_API_KEY di-set)
# ---------------------------------------------------------------------------
_ARJUM_BASE = "https://stock.arjum.com"
_ARJUM_LIMIT = 200  # 200 hari bursa ≈ 10 bulan


def _arjum_fetch_one(code: str, api_key: str, limit: int = _ARJUM_LIMIT) -> pd.DataFrame | None:
    """Ambil OHLCV + value IDR dari Arjum untuk satu kode saham."""
    import json
    import urllib.request

    url = f"{_ARJUM_BASE}/api/history/{code}?limit={limit}&frame=daily"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    rows = data.get("rows")
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume", "value": "Value"})
    needed = ["Open", "High", "Low", "Close", "Volume", "Value"]
    if any(c not in df.columns for c in needed):
        return None
    return df[needed].astype(float).dropna(how="all")


def _arjum_download(tickers: list[str], api_key: str,
                    limit: int = _ARJUM_LIMIT, max_workers: int = 30) -> dict[str, pd.DataFrame]:
    """Download semua ticker dari Arjum API secara paralel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(t: str):
        return t, _arjum_fetch_one(t, api_key, limit)

    result: dict[str, pd.DataFrame] = {}
    print(f"  → Arjum API: {len(tickers)} ticker ({max_workers} worker paralel)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        ok = fail = 0
        for fut in as_completed(futures):
            t, df = fut.result()
            if df is not None and not df.empty:
                result[t.upper()] = df
                ok += 1
            else:
                fail += 1
    print(f"  → Arjum: {ok} berhasil, {fail} tidak ada data / gagal")
    return result


# ---------------------------------------------------------------------------
# Screening universe
# ---------------------------------------------------------------------------
def screen_universe(tickers: list[str], cfg: StaircaseConfig,
                    period: str = "6mo", interval: str = "1d",
                    as_of: str | None = None) -> pd.DataFrame:
    """Screen semua ticker, return DataFrame hasil.

    Prioritas data: Arjum API (ARJUM_API_KEY env var) → yfinance (fallback).
    """
    rows = []
    cutoff = pd.Timestamp(as_of) if as_of else None

    arjum_key = os.environ.get("ARJUM_API_KEY", "").strip()

    if arjum_key:
        codes = [t.replace(".JK", "").upper() for t in tickers]
        print(f"Sumber data: Arjum API ({len(codes)} ticker)...")
        data_map = _arjum_download(codes, arjum_key)
        print(f"Berhasil download: {len(data_map)} ticker")

        def _get_df(ticker: str) -> pd.DataFrame | None:
            return data_map.get(ticker.replace(".JK", "").upper())
    else:
        symbols = [to_yahoo_symbol(t) for t in tickers]
        print(f"Sumber data: yfinance ({len(symbols)} ticker)...")
        data_map = _batch_download(symbols, period, interval)
        print(f"Berhasil download: {len(data_map)} ticker")

        def _get_df(ticker: str) -> pd.DataFrame | None:
            return data_map.get(to_yahoo_symbol(ticker).upper())

    for ticker in tickers:
        raw_df = _get_df(ticker)
        if raw_df is None or raw_df.empty:
            continue
        try:
            if cutoff is not None:
                idx_ts = (raw_df.index.tz_localize(None)
                          if raw_df.index.tz is not None else raw_df.index)
                raw_df = raw_df[idx_ts <= cutoff]
                if raw_df.empty:
                    continue

            result = score_stock(raw_df, cfg)
            if result is None:
                continue

            result["ticker"] = ticker.upper()
            result["as_of"]  = raw_df.index[-1].date().isoformat()
            rows.append(result)

        except Exception as exc:
            print(f"[error] {ticker}: {exc}")

    if not rows:
        return pd.DataFrame()

    cols = [
        "ticker", "as_of", "score", "score_raw",
        "A", "B", "C", "D",
        "num_cycles", "avg_step_gain_pct", "last_step_gain_pct",
        "avg_vol_ratio", "avg_land_range_pct", "pattern_span_days",
        "last_close", "ma5", "ma20", "avg_value_b",
    ]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]] \
              .sort_values("score", ascending=False) \
              .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Format laporan Telegram
# ---------------------------------------------------------------------------
def format_telegram(results: pd.DataFrame, total_screened: int,
                    min_score: float) -> str:
    """Format hasil screening ke pesan Telegram (Markdown)."""
    from datetime import date

    today = date.today().strftime("%d %b %Y")
    n     = len(results)

    header = (
        f"📊 *Laporan Tangga Screener IDX*\n"
        f"Data: {today} | "
        f"{n} setup skor ≥{int(min_score)} (dari {total_screened} saham diproses)\n"
        f"{'─' * 30}\n\n"
    )

    if results.empty:
        return header + "_Tidak ada setup yang memenuhi syarat hari ini._"

    body = ""
    for _, r in results.iterrows():
        body += (
            f"*{r['ticker']}* — skor {int(r['score'])} (raw {int(r['score_raw'])})\n"
            f"A:{int(r['A'])} B:{int(r['B'])} C:{int(r['C'])} D:{int(r['D'])}\n"
            f"Close {r['last_close']:,.0f} | MA5 {r['ma5']:,.0f} | MA20 {r['ma20']:,.0f}\n"
            f"Tangga: {int(r['num_cycles'])} siklus | "
            f"Avg step +{r['avg_step_gain_pct']}% | "
            f"Vol ratio {r['avg_vol_ratio']}×\n"
            f"Span: {int(r['pattern_span_days'])} hari | "
            f"Likuiditas: {r['avg_value_b']:.1f}M/hr\n\n"
        )

    return header + body


def send_telegram(text: str,
                  bot_token: str | None = None,
                  chat_id: str | None = None) -> bool:
    """Kirim pesan ke Telegram. Return True jika berhasil."""
    import urllib.request, urllib.parse, json

    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    cid   = chat_id   or os.environ.get("TELEGRAM_CHAT_ID",   "")
    if not token or not cid:
        print("[telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diset, skip.")
        return False

    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id":    cid,
        "text":       text,
        "parse_mode": "Markdown",
    }).encode()

    try:
        with urllib.request.urlopen(url, data=data, timeout=15) as resp:
            body = json.loads(resp.read())
            if body.get("ok"):
                print("[telegram] Pesan terkirim.")
                return True
            print(f"[telegram] Gagal: {body}")
            return False
    except Exception as exc:
        print(f"[telegram] Error: {exc}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staircase screener IDX — cari pola tangga (landing→step berulang)."
    )
    parser.add_argument("--tickers",      default=None,
        help="Daftar ticker dipisah koma, mis. NASI,BBCA,TLKM (suffix .JK opsional)")
    parser.add_argument("--tickers-file", default=None,
        help="File .txt berisi satu ticker per baris")
    parser.add_argument("--period",   default="6mo",
        help="Rentang data historis yfinance: 3mo/6mo/1y (default 6mo)")
    parser.add_argument("--interval", default="1d",
        help="Interval candle (default 1d)")
    parser.add_argument("--min-score", type=float, default=50.0,
        help="Filter skor minimum (default 50)")
    parser.add_argument("--top",  type=int, default=20,
        help="Jumlah baris teratas yang ditampilkan (default 20)")
    parser.add_argument("--output", default=None,
        help="Path CSV untuk simpan semua hasil")
    parser.add_argument("--as-of", default=None,
        help="YYYY-MM-DD: evaluasi seolah itu 'hari ini' (untuk backtest/validasi)")
    parser.add_argument("--telegram", action="store_true",
        help="Kirim hasil ke Telegram (butuh env TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
    parser.add_argument("--min-value",  type=float, default=None,
        help="Min avg value/hari dalam MILIAR IDR (default 1.0). Set 0 untuk disable.")
    parser.add_argument("--min-cycles", type=int, default=None,
        help="Min jumlah siklus landing→step yang dibutuhkan (default 2)")
    parser.add_argument("--min-step-gain", type=float, default=None,
        help="Min gain per step dalam persen (default 4.0)")

    args = parser.parse_args()

    # ── Parse tickers ─────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        lines   = Path(args.tickers_file).read_text().splitlines()
        tickers = [ln.strip().upper() for ln in lines
                   if ln.strip() and not ln.strip().startswith("#")]
    else:
        parser.error("Wajib isi salah satu: --tickers atau --tickers-file")
        return

    if not tickers:
        print("Tidak ada ticker ditemukan.")
        return

    # ── Konfigurasi (override dari CLI kalau ada) ─────────────────
    cfg = StaircaseConfig()
    if args.min_value   is not None: cfg.min_avg_value_b    = args.min_value
    if args.min_cycles  is not None: cfg.min_cycles         = args.min_cycles
    if args.min_step_gain is not None:
        cfg.step_min_gain_pct = args.min_step_gain / 100.0

    print(
        f"Mengscreen {len(tickers)} ticker "
        f"(period={args.period}, lookback={cfg.lookback_days} hari, "
        f"min_cycles={cfg.min_cycles}, step≥{cfg.step_min_gain_pct*100:.0f}%)…"
    )

    results = screen_universe(tickers, cfg, period=args.period,
                               interval=args.interval, as_of=args.as_of)

    if results.empty:
        msg = "Tidak ada hasil valid."
        print(msg)
        if args.telegram:
            send_telegram(msg)
        return

    filtered = results[results["score"] >= args.min_score].copy()

    print(f"\n=== {len(filtered)} saham dengan skor ≥{args.min_score} "
          f"(dari {len(results)} berhasil diproses) ===\n")
    print(filtered.head(args.top).to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSemua hasil disimpan ke: {args.output}")

    if args.telegram:
        msg = format_telegram(filtered.head(args.top), len(results), args.min_score)
        send_telegram(msg)


if __name__ == "__main__":
    main()
