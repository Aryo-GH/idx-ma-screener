#!/usr/bin/env python3
"""
idx_ma_setup_screener.py

Screener + sistem skoring 0-100 untuk framework "MA Breakout-Retest"
(Setup A/B dari dokumen framework-ma-breakout-retest.md), pakai yfinance
sebagai sumber data historis (gratis, tidak perlu API key).

Konsep skor (total 100):
  A. Struktur breakout dari MA panjang (MA20)       -> bobot 25
  B. Kualitas koreksi setelah breakout (kecil/doji) -> bobot 25
  C. Posisi harga vs MA5 ("nempel")                 -> bobot 30
  D. Konfirmasi tren & momentum (MA5>MA20, StochRSI) -> bobot 20

DATA: YFINANCE
--------------
Saham IDX di Yahoo Finance pakai suffix ".JK" (mis. BBCA -> BBCA.JK).
Ticker yang kamu kasih (dengan atau tanpa .JK) otomatis dinormalisasi.

Dependency: `pip install yfinance --break-system-packages` (kalau belum ada).

CONTOH PAKAI
------------
    # Beberapa ticker langsung
    python idx_ma_setup_screener.py --tickers BBCA,TOWR,ISAT,COCO --min-score 60

    # Screening banyak ticker dari file (satu ticker per baris, boleh pakai list
    # universe IDX yang sudah kamu punya dari idx_closing_spike_scanner.py dll)
    python idx_ma_setup_screener.py --tickers-file idx_universe.txt --min-score 65 --top 30 --output hasil.csv

    # Kalau screening ratusan ticker sekaligus, kasih jeda supaya tidak kena
    # rate-limit dari Yahoo Finance:
    python idx_ma_setup_screener.py --tickers-file idx_universe.txt --sleep 0.5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Konfigurasi bobot & threshold -- silakan di-tuning setelah divalidasi
# terhadap contoh-contoh setup nyata bersama teman kamu.
# ---------------------------------------------------------------------------
@dataclass
class ScoringConfig:
    WEIGHT_A: float = 25.0
    WEIGHT_B: float = 25.0
    WEIGHT_C: float = 30.0
    WEIGHT_D: float = 20.0

    MIN_HISTORY: int = 45  # minimal jumlah candle supaya semua indikator valid

    # Komponen A: body/volume candle breakout vs rata-rata 20 candle
    MIN_BODY_MULT: float = 1.3
    STRONG_BODY_MULT: float = 2.5
    MIN_VOL_MULT: float = 1.3
    STRONG_VOL_MULT: float = 2.5

    # Komponen B: body/volume candle koreksi dibandingkan candle breakout
    CORR_BODY_GOOD: float = 0.5
    CORR_BODY_BAD: float = 1.2
    CORR_VOL_GOOD: float = 0.5
    CORR_VOL_BAD: float = 1.0
    # Gate B -- koreksi disqualified kalau:
    # (1) ada candle merah dengan body > X kali body breakout ("full candle merah")
    CORR_RED_BODY_MAX: float = 0.5   # body merah > 50% body breakout = jelek
    # (2) close jatuh lebih dari X% di bawah MA5 ("terlalu jauh dari MA5")
    CORR_MAX_BELOW_MA5_PCT: float = 7.0  # toleransi 7% di bawah MA5

    # Komponen C: jarak (%) close vs MA5
    MA5_IDEAL_ABOVE_PCT: float = 2.0
    MA5_EXTENDED_PCT: float = 8.0
    MA5_BELOW_FAIL_PCT: float = 3.0

    # Komponen D: level Stoch RSI %K (14,14,3,3)
    STOCH_OB_START: float = 70.0

    # Staleness penalty: setup ideal terjadi dalam ~5 candle setelah breakout.
    # Setelah STALE_CANDLES, skor mulai diturunkan. Setelah MAX_CANDLES,
    # skor total dikunci maksimal MAX_STALE_SCORE.
    STALE_CANDLES: int = 7        # mulai kena penalty setelah sekian candle
    MAX_CANDLES: int = 15         # di atas ini skor dikunci <= MAX_STALE_SCORE
    MAX_STALE_SCORE: float = 60.0 # batas atas skor kalau breakout sudah terlalu lama

    # Liquidity filter -- saham yang tidak memenuhi threshold ini di-skip sepenuhnya.
    # Dihitung dari rata-rata 20 candle terakhir. Set 0 untuk disable.
    MIN_AVG_VALUE: float = 5_000_000_000  # avg value traded/hari (IDR), default 5 miliar
    MIN_AVG_VOLUME: float = 0             # avg volume/hari (lembar), default off
    MIN_PRICE: float = 100                # min last close (IDR), default 100


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def linear_score(x: float, x0: float, x1: float) -> float:
    """0 di x<=x0, naik linear ke 1 di x>=x1 (atau turun kalau x1<x0)."""
    if x0 == x1:
        return 1.0 if x >= x0 else 0.0
    t = (x - x0) / (x1 - x0)
    return clamp(t)


def to_yahoo_symbol(ticker: str) -> str:
    """Normalisasi ticker IDX ke format Yahoo Finance (suffix .JK)."""
    t = ticker.strip().upper()
    if not t.endswith(".JK"):
        t = f"{t}.JK"
    return t


# ---------------------------------------------------------------------------
# Data loading (yfinance)
# ---------------------------------------------------------------------------
def load_ohlc_yf(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Ambil data OHLCV dari yfinance untuk satu ticker IDX.

    Return DataFrame ber-index Date, kolom: Open, High, Low, Close, Volume.
    """
    import yfinance as yf  # import lokal supaya modul ini tetap bisa di-lint/dites tanpa yfinance terpasang

    symbol = to_yahoo_symbol(ticker)
    hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError(f"Tidak ada data dari yfinance untuk {symbol}")
    hist.index.name = "Date"
    hist = hist.rename(columns=str.title)  # jaga-jaga kalau kolom lowercase
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in hist.columns]
    if missing:
        raise ValueError(f"Kolom hilang {missing} dari data yfinance {symbol}")
    return hist[required].astype(float)


# ---------------------------------------------------------------------------
# Indikator
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(series: pd.Series, rsi_period=14, stoch_period=14,
               k_smooth=3, d_smooth=3):
    rsi_vals = _rsi(series, rsi_period)
    min_rsi = rsi_vals.rolling(stoch_period).min()
    max_rsi = rsi_vals.rolling(stoch_period).max()
    stoch = (rsi_vals - min_rsi) / (max_rsi - min_rsi).replace(0, pd.NA) * 100
    k = stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["AvgBody20"] = df["Body"].rolling(20).mean()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["Value"] = df["Close"] * df["Volume"]           # value traded per candle (IDR)
    df["AvgValue20"] = df["Value"].rolling(20).mean()  # rata-rata 20 candle
    df["StochK"], df["StochD"] = _stoch_rsi(df["Close"])
    return df


def check_liquidity(df: pd.DataFrame, cfg: ScoringConfig) -> tuple[bool, str]:
    """Return (lolos, keterangan). Kalau False, saham di-skip dari hasil."""
    last = df.iloc[-1]
    close = last["Close"]
    avg_vol = last["AvgVol20"]
    avg_val = last["AvgValue20"]

    reasons = []

    if cfg.MIN_PRICE > 0 and (pd.isna(close) or close < cfg.MIN_PRICE):
        reasons.append(f"harga {close:.0f} < min {cfg.MIN_PRICE:.0f}")

    if cfg.MIN_AVG_VOLUME > 0 and (pd.isna(avg_vol) or avg_vol < cfg.MIN_AVG_VOLUME):
        reasons.append(f"avg vol {avg_vol/1e6:.1f}jt lembar < min {cfg.MIN_AVG_VOLUME/1e6:.0f}jt")

    if cfg.MIN_AVG_VALUE > 0 and (pd.isna(avg_val) or avg_val < cfg.MIN_AVG_VALUE):
        val_b = avg_val / 1e9 if pd.notna(avg_val) else 0
        min_b = cfg.MIN_AVG_VALUE / 1e9
        reasons.append(f"avg value {val_b:.2f}M < min {min_b:.1f}M")

    if reasons:
        return False, "ILLIQUID: " + "; ".join(reasons)

    val_b = avg_val / 1e9 if pd.notna(avg_val) else 0
    return True, f"liquid (avg {val_b:.1f}M/hari, close {close:.0f})"


# ---------------------------------------------------------------------------
# Deteksi breakout
# ---------------------------------------------------------------------------
def find_breakout(df: pd.DataFrame, lookback: int, cfg: ScoringConfig):
    """Cari index candle breakout paling baru dalam `lookback` candle terakhir.

    Syarat breakout yang valid (framework "satu full candle hijau besar, lalu retrace tertahan"):
    - candle HIJAU (close > open) -- bukan candle merah/doji
    - close menembus ke atas MA20 (crossover dari <= ke >)
    - body candle >= MIN_BODY_MULT x rata-rata body 20 candle
    - volume >= MIN_VOL_MULT x rata-rata volume 20 candle
    - MA20 tidak sedang turun tajam saat breakout (slope MA20 10 candle >= -2%)
      --> filter saham yang "mantul" sementara di tengah downtrend panjang

    Return None kalau tidak ketemu.
    """
    MA20_SLOPE_LOOKBACK = 10  # candle lookback untuk cek arah MA20
    n = len(df)
    # 20 = jumlah minimum candle sebelumnya supaya MA20/AvgBody20/AvgVol20 valid,
    # BUKAN cfg.MIN_HISTORY (itu syarat total panjang data, beda tujuan)
    start = max(20, n - lookback)
    for i in range(n - 1, start - 1, -1):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(row["MA20"]) or pd.isna(row["AvgBody20"]) or pd.isna(row["AvgVol20"]):
            continue

        # 1. Harus candle hijau (close > open)
        green_candle = row["Close"] > row["Open"]

        # 2. Crossover: kemarin di bawah MA20, hari ini di atas
        crossed = prev["Close"] <= prev["MA20"] and row["Close"] > row["MA20"]

        # 3. Body dan volume besar
        body = row["Close"] - row["Open"]  # positif karena green_candle
        body_ok = row["AvgBody20"] > 0 and body >= cfg.MIN_BODY_MULT * row["AvgBody20"]
        vol_ok = row["AvgVol20"] > 0 and row["Volume"] >= cfg.MIN_VOL_MULT * row["AvgVol20"]

        # 4. MA20 tidak dalam downtrend tajam (slope >= -2% dari 10 candle lalu)
        if i >= MA20_SLOPE_LOOKBACK:
            ma20_ago = df.iloc[i - MA20_SLOPE_LOOKBACK]["MA20"]
            ma20_slope_ok = pd.isna(ma20_ago) or row["MA20"] >= ma20_ago * 0.98
        else:
            ma20_slope_ok = True

        if crossed and body_ok and vol_ok and green_candle and ma20_slope_ok:
            return i
    return None


# ---------------------------------------------------------------------------
# Komponen skoring (A, B, C, D)
# ---------------------------------------------------------------------------
def score_breakout_structure(df, breakout_idx, cfg: ScoringConfig):
    if breakout_idx is None:
        last = df.iloc[-1]
        if pd.notna(last["MA20"]) and last["Close"] > last["MA20"]:
            return cfg.WEIGHT_A * 0.3, "Di atas MA20 tapi tanpa breakout impulsif yang jelas"
        return 0.0, "Belum breakout dari MA20 (masih di bawah/menekan)"

    row = df.iloc[breakout_idx]
    body = row["Close"] - row["Open"]  # positif (green candle sudah dijamin find_breakout)
    body_ratio = body / row["AvgBody20"] if row["AvgBody20"] > 0 else 0.0
    vol_ratio = row["Volume"] / row["AvgVol20"] if row["AvgVol20"] > 0 else 0.0
    body_q = linear_score(body_ratio, cfg.MIN_BODY_MULT, cfg.STRONG_BODY_MULT)
    vol_q = linear_score(vol_ratio, cfg.MIN_VOL_MULT, cfg.STRONG_VOL_MULT)
    quality = 0.5 * body_q + 0.5 * vol_q
    note = f"Breakout body={body_ratio:.2f}x avg, vol={vol_ratio:.2f}x avg ({row.name.date()})"
    return quality * cfg.WEIGHT_A, note


def score_correction_quality(df, breakout_idx, cfg: ScoringConfig):
    if breakout_idx is None:
        return 0.0, "N/A (belum ada breakout)"
    corr = df.iloc[breakout_idx + 1:]
    if len(corr) == 0:
        return cfg.WEIGHT_B * 0.5, "Belum ada candle koreksi (baru saja breakout)"

    breakout_row = df.iloc[breakout_idx]
    breakout_body = breakout_row["Close"] - breakout_row["Open"]  # green candle, positif

    # Gate 1: tidak boleh ada "full candle merah" besar saat koreksi.
    # Full candle merah = candle merah dengan body > CORR_RED_BODY_MAX * body breakout.
    big_red_threshold = breakout_body * cfg.CORR_RED_BODY_MAX
    for ts, cr in corr.iterrows():
        red_body = cr["Open"] - cr["Close"]  # positif kalau merah, negatif kalau hijau
        if red_body > big_red_threshold:
            return 0.0, f"Koreksi ada full candle merah besar pada {ts.date()} (body {red_body:.0f} vs threshold {big_red_threshold:.0f})"

    # Gate 2: harga tidak boleh jatuh terlalu jauh di bawah MA5 ("sekitar MA5").
    # Boleh sedikit di bawah MA5, tapi tidak lebih dari CORR_MAX_BELOW_MA5_PCT.
    for ts, cr in corr.iterrows():
        if pd.isna(cr["MA5"]) or cr["MA5"] == 0:
            continue
        floor = cr["MA5"] * (1 - cfg.CORR_MAX_BELOW_MA5_PCT / 100)
        if cr["Close"] < floor:
            return 0.0, f"Koreksi terlalu jauh di bawah MA5 pada {ts.date()} (close {cr['Close']:.0f} vs MA5 {cr['MA5']:.0f})"
    breakout_vol = breakout_row["Volume"]
    avg_corr_body = (corr["Close"] - corr["Open"]).abs().mean()
    avg_corr_vol = corr["Volume"].mean()

    body_ratio = avg_corr_body / breakout_body if breakout_body > 0 else 1.0
    vol_ratio = avg_corr_vol / breakout_vol if breakout_vol > 0 else 1.0
    body_q = 1 - linear_score(body_ratio, cfg.CORR_BODY_GOOD, cfg.CORR_BODY_BAD)
    vol_q = 1 - linear_score(vol_ratio, cfg.CORR_VOL_GOOD, cfg.CORR_VOL_BAD)
    quality = clamp(0.5 * body_q + 0.5 * vol_q)
    note = f"Koreksi tertahan di atas MA20, body={body_ratio:.2f}x, vol={vol_ratio:.2f}x breakout ({len(corr)} candle)"
    return quality * cfg.WEIGHT_B, note


def score_ma5_position(last_row, cfg: ScoringConfig):
    ma5, close = last_row["MA5"], last_row["Close"]
    if pd.isna(ma5) or ma5 == 0:
        return 0.0, "MA5 tidak tersedia"
    dist_pct = (close - ma5) / ma5 * 100
    if dist_pct >= 0:
        if dist_pct <= cfg.MA5_IDEAL_ABOVE_PCT:
            quality = 1.0
        else:
            quality = 1.0 - linear_score(dist_pct, cfg.MA5_IDEAL_ABOVE_PCT, cfg.MA5_EXTENDED_PCT)
    else:
        quality = 1.0 - linear_score(abs(dist_pct), 0.0, cfg.MA5_BELOW_FAIL_PCT)
    quality = clamp(quality)
    return quality * cfg.WEIGHT_C, f"Close {dist_pct:+.2f}% vs MA5"


def score_trend_momentum(df, cfg: ScoringConfig):
    last = df.iloc[-1]
    trend_ok = pd.notna(last["MA5"]) and pd.notna(last["MA20"]) and last["MA5"] > last["MA20"]
    trend_score = cfg.WEIGHT_D * 0.5 if trend_ok else 0.0

    k, d = last["StochK"], last["StochD"]
    if pd.isna(k) or pd.isna(d):
        momentum_score = cfg.WEIGHT_D * 0.25
        note_k = "StochRSI N/A"
    else:
        bearish_cross = k < d  # %K sudah turun di bawah %D -> momentum mulai melemah/rollover
        if k >= cfg.STOCH_OB_START:
            # Momentum kuat, TERMASUK yang sangat overbought (>90) -- ini BUKAN sinyal
            # jelek selama %K belum cross turun di bawah %D. Overbought yang persisten/
            # naik justru ciri breakout kuat, bukan alarm reversal. Hanya diturunkan
            # kalau sudah ada bearish crossover yang menandakan momentum mulai habis.
            mom_q = 0.4 if bearish_cross else 1.0
        else:
            mom_q = linear_score(k, 30.0, cfg.STOCH_OB_START)
            if bearish_cross:
                mom_q *= 0.5
        momentum_score = mom_q * cfg.WEIGHT_D * 0.5
        note_k = f"StochK={k:.1f}, StochD={d:.1f}{' (bearish cross)' if bearish_cross else ''}"

    note = f"MA5>MA20={trend_ok}, {note_k}"
    return trend_score + momentum_score, note


# ---------------------------------------------------------------------------
# Staleness penalty
# ---------------------------------------------------------------------------
def apply_staleness_penalty(total: float, candles_since_breakout: int | None,
                             cfg: ScoringConfig) -> tuple[float, str]:
    """Turunkan skor kalau breakout sudah terlalu lama.

    - 0..STALE_CANDLES   : tidak ada penalty
    - STALE_CANDLES..MAX_CANDLES : penalty linear (skor dikali faktor turun)
    - > MAX_CANDLES      : skor dikunci maks MAX_STALE_SCORE
    """
    if candles_since_breakout is None:
        return total, ""

    n = candles_since_breakout
    if n <= cfg.STALE_CANDLES:
        return total, f"fresh ({n}c sejak breakout)"

    if n >= cfg.MAX_CANDLES:
        penalized = min(total, cfg.MAX_STALE_SCORE)
        return round(penalized, 1), f"STALE ({n}c sejak breakout, skor dikunci <={cfg.MAX_STALE_SCORE:.0f})"

    # Linear: dari 1.0 di STALE_CANDLES turun ke MAX_STALE_SCORE/100 di MAX_CANDLES
    t = (n - cfg.STALE_CANDLES) / (cfg.MAX_CANDLES - cfg.STALE_CANDLES)
    floor_ratio = cfg.MAX_STALE_SCORE / 100.0
    factor = 1.0 - t * (1.0 - floor_ratio)
    penalized = total * factor
    return round(penalized, 1), f"agak stale ({n}c sejak breakout, penalty {factor:.2f}x)"


# ---------------------------------------------------------------------------
# Skor total per saham
# ---------------------------------------------------------------------------
def score_stock(raw_df: pd.DataFrame, cfg: ScoringConfig, lookback: int):
    if len(raw_df) < cfg.MIN_HISTORY:
        return None  # data terlalu pendek

    df = compute_indicators(raw_df)

    # Liquidity gate -- return None kalau tidak liquid (akan di-skip)
    liquid_ok, liquid_note = check_liquidity(df, cfg)
    if not liquid_ok:
        return None  # caller akan print skip message

    breakout_idx = find_breakout(df, lookback, cfg)

    a, note_a = score_breakout_structure(df, breakout_idx, cfg)
    b, note_b = score_correction_quality(df, breakout_idx, cfg)
    c, note_c = score_ma5_position(df.iloc[-1], cfg)
    d, note_d = score_trend_momentum(df, cfg)

    raw_total = a + b + c + d

    # Hitung candle sejak breakout
    candles_since = (len(df) - 1 - breakout_idx) if breakout_idx is not None else None
    total, stale_note = apply_staleness_penalty(raw_total, candles_since, cfg)

    last = df.iloc[-1]
    avg_val_b = round(last["AvgValue20"] / 1e9, 2) if pd.notna(last["AvgValue20"]) else None
    notes_parts = [note_a, note_b, note_c, note_d]
    if stale_note:
        notes_parts.append(stale_note)

    return {
        "score_total": total,
        "score_raw": round(raw_total, 1),
        "score_A_breakout": round(a, 1),
        "score_B_correction": round(b, 1),
        "score_C_ma5_position": round(c, 1),
        "score_D_trend_momentum": round(d, 1),
        "candles_since_breakout": candles_since,
        "avg_daily_value_B": avg_val_b,   # dalam miliar IDR
        "breakout_date": df.index[breakout_idx].date().isoformat() if breakout_idx is not None else None,
        "last_close": round(last["Close"], 2),
        "ma5": round(last["MA5"], 2) if pd.notna(last["MA5"]) else None,
        "ma20": round(last["MA20"], 2) if pd.notna(last["MA20"]) else None,
        "notes": " | ".join(notes_parts),
    }


# ---------------------------------------------------------------------------
# Screening seluruh universe (via yfinance)
# ---------------------------------------------------------------------------
def screen_universe(tickers: list[str], lookback: int, cfg: ScoringConfig,
                     period: str = "6mo", interval: str = "1d",
                     sleep: float = 0.0, as_of: str | None = None) -> pd.DataFrame:
    rows = []
    cutoff = pd.Timestamp(as_of) if as_of else None
    for ticker in tickers:
        try:
            raw_df = load_ohlc_yf(ticker, period=period, interval=interval)
            if cutoff is not None:
                raw_df = raw_df[raw_df.index.tz_localize(None) <= cutoff] if raw_df.index.tz is not None \
                    else raw_df[raw_df.index <= cutoff]
                if raw_df.empty:
                    print(f"[skip] {ticker}: tidak ada data pada/sebelum {as_of}")
                    continue
            result = score_stock(raw_df, cfg, lookback)
            if result is None:
                # Bisa karena data pendek ATAU tidak liquid -- cek dulu panjang data
                if len(raw_df) < cfg.MIN_HISTORY:
                    print(f"[skip] {ticker}: data kurang ({len(raw_df)} candle)")
                else:
                    # Hitung ulang untuk dapat pesan liquidity
                    df_tmp = compute_indicators(raw_df)
                    _, liq_msg = check_liquidity(df_tmp, cfg)
                    print(f"[skip] {ticker}: {liq_msg}")
                continue
            result["ticker"] = ticker.upper()
            result["as_of"] = raw_df.index[-1].date().isoformat()
            rows.append(result)
        except Exception as exc:  # noqa: BLE001 -- 1 ticker error tidak boleh menghentikan seluruh screening
            print(f"[error] {ticker}: {exc}")
        if sleep > 0:
            time.sleep(sleep)

    if not rows:
        return pd.DataFrame()

    cols_order = [
        "ticker", "as_of", "score_total", "score_raw",
        "score_A_breakout", "score_B_correction",
        "score_C_ma5_position", "score_D_trend_momentum",
        "candles_since_breakout", "avg_daily_value_B",
        "last_close", "ma5", "ma20", "breakout_date", "notes",
    ]
    out = pd.DataFrame(rows)
    return out[cols_order].sort_values("score_total", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Screener & scoring 0-100 untuk framework MA Breakout-Retest (Setup A/B), data via yfinance."
    )
    parser.add_argument("--tickers", default=None,
                         help="Daftar ticker dipisah koma, mis. BBCA,TOWR,ISAT (suffix .JK opsional)")
    parser.add_argument("--tickers-file", default=None,
                         help="File .txt berisi satu ticker per baris (baris kosong / diawali # diabaikan)")
    parser.add_argument("--lookback", type=int, default=15,
                         help="Jumlah candle terakhir untuk mencari breakout (default 15)")
    parser.add_argument("--period", default="6mo",
                         help="Rentang data historis yfinance, mis. 3mo/6mo/1y (default 6mo)")
    parser.add_argument("--interval", default="1d",
                         help="Interval candle yfinance (default 1d)")
    parser.add_argument("--min-score", type=float, default=0.0,
                         help="Hanya tampilkan hasil dengan skor >= nilai ini")
    parser.add_argument("--top", type=int, default=30,
                         help="Jumlah baris teratas yang ditampilkan di layar (default 30)")
    parser.add_argument("--output", default=None,
                         help="Path CSV untuk simpan SEMUA hasil (opsional)")
    parser.add_argument("--sleep", type=float, default=0.0,
                         help="Jeda (detik) antar request yfinance -- pakai 0.3-1.0 kalau screening banyak ticker "
                              "supaya tidak kena rate-limit")
    parser.add_argument("--as-of", default=None,
                         help="Format YYYY-MM-DD. Evaluasi seolah-olah tanggal itu 'hari ini' (data setelahnya "
                              "dipotong) -- untuk validasi/backtest screener terhadap contoh setup historis. "
                              "Pakai --period yang cukup panjang (mis. 1y) supaya tanggal itu tercakup.")
    parser.add_argument("--min-value", type=float, default=None,
                         help="Min avg value traded/hari dalam MILIAR IDR (default 1.0 = 1 miliar). "
                              "Contoh: --min-value 2.5 untuk filter >= 2.5 miliar/hari. Set 0 untuk disable.")
    parser.add_argument("--min-price", type=float, default=None,
                         help="Min harga last close dalam IDR (default 50). Set 0 untuk disable.")
    parser.add_argument("--min-volume", type=float, default=None,
                         help="Min avg volume/hari dalam JUTA lembar saham (default 0 = off). "
                              "Contoh: --min-volume 1 untuk filter >= 1 juta lembar/hari.")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        lines = Path(args.tickers_file).read_text().splitlines()
        tickers = [ln.strip().upper() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    else:
        parser.error("Wajib isi salah satu: --tickers atau --tickers-file")
        return

    if not tickers:
        print("Tidak ada ticker ditemukan (cek --tickers / --tickers-file).")
        return

    cfg = ScoringConfig()
    # Override liquidity threshold dari CLI kalau ada
    if args.min_value is not None:
        cfg.MIN_AVG_VALUE = args.min_value * 1_000_000_000
    if args.min_price is not None:
        cfg.MIN_PRICE = args.min_price
    if args.min_volume is not None:
        cfg.MIN_AVG_VOLUME = args.min_volume * 1_000_000

    liq_info = (f"value>={cfg.MIN_AVG_VALUE/1e9:.1f}M/hari, "
                f"price>={cfg.MIN_PRICE:.0f}, "
                f"vol>={cfg.MIN_AVG_VOLUME/1e6:.0f}jt/hari")
    print(f"Mengambil & menskor {len(tickers)} ticker dari yfinance "
          f"(period={args.period}, interval={args.interval})...")
    print(f"Liquidity filter: {liq_info}")
    results = screen_universe(tickers, args.lookback, cfg, period=args.period,
                               interval=args.interval, sleep=args.sleep, as_of=args.as_of)

    if results.empty:
        print("Tidak ada hasil valid.")
        return

    filtered = results[results["score_total"] >= args.min_score]
    display_cols = [c for c in results.columns if c != "notes"]

    print(f"\n=== {len(filtered)} saham dengan skor >= {args.min_score} (dari {len(results)} berhasil diproses) ===\n")
    print(filtered[display_cols].head(args.top).to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSemua hasil (termasuk catatan detail) disimpan ke: {args.output}")


if __name__ == "__main__":
    main()
