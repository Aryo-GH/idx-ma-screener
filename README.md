# IDX MA Setup Screener — GitHub Actions Nightly

Paket ini menjalankan `idx_ma_setup_screener.py` setiap malam lewat GitHub
Actions (gratis untuk repo, termasuk repo privat sampai batas menit bulanan
yang cukup besar), lalu kirim ringkasan hasil ke Telegram.

## Struktur folder

```
.github/workflows/nightly_screener.yml   <- jadwal & langkah-langkah otomatisnya
scripts/idx_ma_setup_screener.py         <- screener + sistem skoring (Setup A/B)
scripts/send_telegram_report.py          <- format & kirim hasil ke Telegram
data/idx_universe.txt                    <- daftar ticker yang di-screening (GANTI ISINYA)
results/latest.csv                       <- hasil run terakhir (dibuat otomatis)
results/history/YYYY-MM-DD.csv           <- arsip harian (dibuat otomatis)
```

## Setup (sekali saja)

### 1. Push folder ini ke repo GitHub kamu
Bisa repo baru atau ditambahkan ke repo yang sudah ada (mis. tempat kamu
simpan `idx_closing_spike_scanner.py` dkk).

### 2. Ganti isi `data/idx_universe.txt`
Isi dengan daftar ticker lengkap yang mau di-screening tiap malam (satu per
baris). Kalau kamu screening ratusan ticker, tambahkan `--sleep 0.5` atau
lebih di workflow supaya tidak kena rate-limit dari Yahoo Finance.

### 3. Buat Telegram bot & dapatkan token + chat id

1. Chat ke **@BotFather** di Telegram → `/newbot` → ikuti instruksinya →
   kamu akan dapat **bot token** (format `123456:ABC-DEF...`).
2. Chat (kirim pesan apa saja) ke bot yang baru kamu buat.
3. Buka di browser:
   `https://api.telegram.org/bot<TOKEN_KAMU>/getUpdates`
   → cari field `"chat":{"id": ...}` di hasil JSON-nya, itu **chat id** kamu.
   (Kalau hasilnya kosong, pastikan sudah kirim pesan ke bot dulu di langkah 2.)

### 4. Simpan token & chat id sebagai GitHub Secrets

Di repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**:
- `TELEGRAM_BOT_TOKEN` → isi bot token dari langkah 3
- `TELEGRAM_CHAT_ID` → isi chat id dari langkah 3

### 5. (Kalau mau commit hasil balik ke repo) Aktifkan permission write

Workflow ini sudah punya `permissions: contents: write` di file YAML-nya,
biasanya itu cukup. Kalau langkah commit di Actions log gagal karena
permission, cek juga: **Settings → Actions → General → Workflow permissions**
→ pilih **"Read and write permissions"**.

## Jadwal

Default: **22:30 WIB, Senin–Jumat** (`cron: "30 15 * * 1-5"`, GitHub Actions
pakai UTC). Ganti angka di file `.github/workflows/nightly_screener.yml`
kalau mau jam lain — format cron: `menit jam tanggal bulan hari(0=Minggu)`.

## Cara tes manual (tanpa nunggu jadwal malam)

Buka tab **Actions** di repo GitHub → pilih workflow **"IDX MA Setup
Screener - Nightly"** → klik **"Run workflow"**.

## Tuning threshold / bobot skor

Ubah `--min-score` dan `--top` di file workflow YAML sesuai selera. Bobot
komponen skor (A/B/C/D) dan threshold detailnya ada di `ScoringConfig`
di bagian atas `scripts/idx_ma_setup_screener.py`.

## Kalau mau pakai email, bukan Telegram

Ganti langkah "Kirim laporan ke Telegram" di YAML dengan action seperti
`dawidd6/action-send-mail`, dan isi secrets SMTP (host, username, app
password) sesuai provider email kamu (mis. Gmail App Password). Bisa saya
bantu buatkan versinya kalau kamu mau pakai jalur ini juga.
