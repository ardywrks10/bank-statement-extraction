from pathlib import Path

DATA_DIR = Path("data")
TMP_DIR = DATA_DIR / "tmp"
KONVERSI_DIR = DATA_DIR / "hasil_konversi"
MATCHING_DIR = DATA_DIR / "bukti_matching"

# pastikan direktori ada
for d in (TMP_DIR, KONVERSI_DIR, MATCHING_DIR):
    d.mkdir(parents=True, exist_ok=True)
