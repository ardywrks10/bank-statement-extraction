from fastapi import HTTPException
from typing import Dict

from app.extractors.bca import BCAExtractor
from app.extractors.bni import BNIExtractor
from app.extractors.permata import PermataExtractor
from app.extractors.mandiri import MandiriExtractor

# OCR reader global (akan diinisialisasi di main.py lifespan)
EASYOCR_READER = None

STATIC_EXTRACTORS = {
    "bca": BCAExtractor,
    "bni": BNIExtractor,
    "permata": PermataExtractor,
    "mandiri": MandiriExtractor,
}

def set_reader(reader):
    """Dipanggil sekali di startup untuk inject easyocr.Reader"""
    global EASYOCR_READER
    EASYOCR_READER = reader

def get_extractor(bank_name: str):
    """Ambil extractor sesuai nama bank"""
    if EASYOCR_READER is None:
        raise HTTPException(status_code=500, detail="OCR belum siap")

    bank = bank_name.lower()
    cls = STATIC_EXTRACTORS.get(bank)
    if not cls:
        raise HTTPException(status_code=400, detail=f"Bank {bank_name} tidak terdaftar")

    return cls(EASYOCR_READER)
