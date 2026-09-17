#!/usr/bin/env python3
"""
send_telegram_report.py

Kirim hasil screener IDX ke Telegram.
Baca CSV dari hasil idx_ma_setup_screener.py, format jadi pesan ringkas.

Env vars yang dibutuhkan (set di GitHub Secrets):
  TELEGRAM_BOT_TOKEN  -- token dari BotFather
  TELEGRAM_CHAT_ID    -- chat ID tujuan (dapat dari getUpdates)
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------
def tg_send(token: str, chat_id: str, text: str):
    """Kirim pesan ke Telegram, auto-split kalau >4096 karakter."""
    for i in range(0, len(text), 4096):
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text[i : i + 4096],
                "parse_mode": "Markdown",
            },
            timeout=30,
        )
        if not resp.ok:
            print(f"[warn] Telegram error: {resp.status_code} {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Format laporan
# ---------------------------------------------------------------------------
def fmt_date_short(iso_str) -> str:
    """'2026-09-17' -> '17 Sep 2026'"""
    if not iso_str or str(iso_str) in ("", "nan", "None", "NaT"):
        return "?"
    try:
        d = date.fromisoformat(str(iso_str)[:10])
        return d.strftime("%-d %b %Y")  # e.g. "17 Sep 2026"
    except Exception:
        return str(iso_str)[:10]


def fmt_date_breakout(iso_str) -> str:
    """'2026-09-15' -> '15 Sep'"""
    if not iso_str or str(iso_str) in ("", "nan", "None", "NaT"):
        return "?"
    try:
        d = date.fromisoformat(str(iso_str)[:10])
        return d.strftime("%-d %b")
    except Exception:
        return str(iso_str)[:10]


def build_report(df: pd.DataFrame, min_score: float, top: int, run_date: str) -> str:
    filtered = df[df["score_total"] >= min_score].head(top)
    total_scanned = len(df)

    header = (
        f"📊 *Laporan Screener IDX*\n"
        f"Data: {fmt_date_short(run_date)} | "
        f"{len(filtered)} setup skor ≥{min_score:.0f} "
        f"(dari {total_scanned} saham diproses)\n"
        f"{'─' * 30}"
    )

    blocks = [header]

    for _, row in filtered.iterrows():
        ticker    = str(row.get("ticker", "?")).upper()
        score     = int(row.get("score_total", 0))
        score_raw = row.get("score_raw", None)
        a = int(row.get("score_A_breakout", 0))
        b = int(row.get("score_B_correction", 0))
        c = int(row.get("score_C_ma5_position", 0))
        d = int(row.get("score_D_trend_momentum", 0))

        close   = row.get("last_close", None)
        as_of   = row.get("as_of", None)
        ma5     = row.get("ma5", None)
        ma20    = row.get("ma20", None)
        bo_date = row.get("breakout_date", None)
        avg_val = row.get("avg_daily_value_B", None)
        candles = row.get("candles_since_breakout", None)

        # Format values
        close_str = f"{close:.0f}" if close is not None and pd.notna(close) else "?"
        as_of_str = fmt_date_short(as_of)   # tanggal data terakhir (untuk verifikasi)
        ma5_str   = f"{ma5:.0f}"  if ma5  is not None and pd.notna(ma5)  else "?"
        ma20_str  = f"{ma20:.0f}" if ma20 is not None and pd.notna(ma20) else "?"
        bo_str    = fmt_date_breakout(bo_date)

        val_str   = f"{avg_val:.1f}M/hr" if avg_val is not None and pd.notna(avg_val) else "-"
        stale_str = f" | {int(candles)}c lalu" if candles is not None and pd.notna(candles) else ""

        # Penanda kalau skor sudah kena staleness penalty
        penalty_note = ""
        if score_raw is not None and pd.notna(score_raw) and float(score_raw) > score:
            penalty_note = f" _(raw {int(score_raw)})_"

        block = (
            f"\n*{ticker}* — skor *{score}*{penalty_note}\n"
            f"A:{a} B:{b} C:{c} D:{d}\n"
            f"Close *{close_str}* (data {as_of_str}) | MA5 {ma5_str} | MA20 {ma20_str}\n"
            f"Breakout: {bo_str}{stale_str} | Likuiditas: {val_str}"
        )
        blocks.append(block)

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kirim hasil screener IDX ke Telegram."
    )
    parser.add_argument("--csv", required=True, help="Path ke CSV hasil screener")
    parser.add_argument("--min-score", type=float, default=60.0,
                        help="Hanya kirim saham dengan skor >= nilai ini (default 60)")
    parser.add_argument("--top", type=int, default=15,
                        help="Jumlah saham teratas yang dikirim (default 15)")
    args = parser.parse_args()

    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tidak di-set di environment.")
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv)
    except Exception as exc:
        print(f"[error] Gagal baca CSV {args.csv}: {exc}")
        sys.exit(1)

    # as_of dari baris pertama (semua saham di-screen pada hari yang sama)
    run_date = df["as_of"].iloc[0] if "as_of" in df.columns and not df.empty else ""

    if df.empty or df["score_total"].max() < args.min_score:
        tg_send(
            token, chat_id,
            f"📊 Screener IDX ({fmt_date_short(run_date)}): tidak ada setup yang memenuhi syarat hari ini."
        )
        print("Tidak ada saham yang lolos filter, pesan kosong dikirim.")
        return

    msg = build_report(df, args.min_score, args.top, run_date)
    tg_send(token, chat_id, msg)

    n_sent = len(df[df["score_total"] >= args.min_score].head(args.top))
    print(f"Laporan terkirim: {n_sent} saham (data: {run_date})")


if __name__ == "__main__":
    main()
