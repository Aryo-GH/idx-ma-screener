#!/usr/bin/env python3
"""
send_telegram_report.py

Baca hasil screener (CSV output dari idx_ma_setup_screener.py) dan kirim
ringkasan ke Telegram lewat Bot API.

ENV VARS (atau lewat --token / --chat-id):
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

CONTOH:
    python send_telegram_report.py --csv results/latest.csv --min-score 60 --top 15
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import requests

TELEGRAM_MSG_LIMIT = 4096


def format_report(df: pd.DataFrame, min_score: float, top: int) -> list[str]:
    """Ubah DataFrame hasil screener jadi satu atau beberapa teks pesan Telegram
    (dipecah otomatis kalau kepanjangan dari limit 4096 karakter Telegram)."""
    filtered = df[df["score_total"] >= min_score].sort_values("score_total", ascending=False).head(top)

    if filtered.empty:
        return [f"\U0001F4CA *Laporan Screener IDX*\n\nTidak ada saham dengan skor >= {min_score} hari ini."]

    header = f"\U0001F4CA *Laporan Screener IDX* -- {len(filtered)} saham skor >= {min_score:g}"
    blocks = [header]
    for _, row in filtered.iterrows():
        breakout_note = f" | Breakout {row['breakout_date']}" if pd.notna(row.get("breakout_date")) else ""
        blocks.append(
            f"*{row['ticker']}* -- skor {row['score_total']:.0f}\n"
            f"  A:{row['score_A_breakout']:.0f}  B:{row['score_B_correction']:.0f}  "
            f"C:{row['score_C_ma5_position']:.0f}  D:{row['score_D_trend_momentum']:.0f}\n"
            f"  Close {row['last_close']:.0f} | MA5 {row['ma5']:.0f} | MA20 {row['ma20']:.0f}{breakout_note}"
        )

    # Pecah jadi beberapa pesan kalau melebihi batas panjang Telegram
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > TELEGRAM_MSG_LIMIT:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_telegram_message(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Kirim laporan hasil screener IDX ke Telegram")
    parser.add_argument("--csv", required=True, help="Path CSV hasil idx_ma_setup_screener.py")
    parser.add_argument("--min-score", type=float, default=60.0, help="Ambang skor yang ditampilkan (default 60)")
    parser.add_argument("--top", type=int, default=15, help="Jumlah saham teratas yang dikirim (default 15)")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"),
                         help="Bot token (default: env TELEGRAM_BOT_TOKEN)")
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"),
                         help="Chat id tujuan (default: env TELEGRAM_CHAT_ID)")
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diisi (env var atau --token/--chat-id).",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    chunks = format_report(df, args.min_score, args.top)

    for chunk in chunks:
        send_telegram_message(args.token, args.chat_id, chunk)

    print(f"Terkirim {len(chunks)} pesan ke Telegram.")


if __name__ == "__main__":
    main()
