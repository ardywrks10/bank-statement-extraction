### -------- Importing Packages -------- ###
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import pathlib, shutil, pandas as pd, easyocr, os, json, re

from extractors.bca import BCAExtractor
from extractors.bni import BNIExtractor
from extractors.permata import PermataExtractor
from extractors.mandiri import MandiriExtractor
from matching import BankJournalMatcher
from extractors.pipeline import Pipeline as DefaultBaseExtractor
import numpy as np
import pandas as pd
from datetime import date, datetime
from sql_bb_export import export_bb_excel_from_sql

def _to_iso_date(x):
    # 1) Tangani None / NaT / NaN lebih dulu
    try:
        if x is None or pd.isna(x):
            return None
    except Exception:
        # kalau objeknya tidak bisa di-check oleh pd.isna
        if x is None:
            return None

    # 2) Pandas Timestamp (termasuk yang non-NaT)
    if isinstance(x, pd.Timestamp):
        # pd.Timestamp NaT juga akan ketangkap di pd.isna() di atas
        try:
            return x.to_pydatetime().strftime("%Y-%m-%d")
        except Exception:
            return None

    # 3) numpy datetime64
    if isinstance(x, np.datetime64):
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        try:
            return dt.to_pydatetime().strftime("%Y-%m-%d")
        except Exception:
            return None

    # 4) Python datetime/date
    if isinstance(x, (datetime, date)):
        try:
            return x.strftime("%Y-%m-%d")
        except Exception:
            return None

    # 5) String mentah (biarkan apa adanya kecuali "-" / "")
    if isinstance(x, str):
        s = x.strip()
        return None if s in {"", "-"} else s

    # 6) Tipe lain -> anggap tidak ada tanggal
    return None


def _snake(s: str) -> str:
    # Ubah label-lable "Tanggal (BB)" → "tanggal_bb", "No Voucher" → "no_voucher"
    s = s.strip().lower()
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    return s

# mapping kolom transaksi → kunci json (agar urutan konsisten)
TRANSACTION_KEY_ORDER = [
    "tanggal_bb","no_voucher","debit_bb","kredit_bb","saldo_bb",
    "tanggal_rk","debit_rk","kredit_rk","saldo_rk",
    "debit_bb_kredit_rk","kredit_bb_debit_rk",
    "status","id","catatan"
]

TRANSACTION_COL_MAP = {
    "Tanggal (BB)": "tanggal_bb",
    "No Voucher": "no_voucher",
    "Debit (BB)": "debit_bb",
    "Kredit (BB)": "kredit_bb",
    "Saldo (BB)": "saldo_bb",
    "Tanggal (RK)": "tanggal_rk",
    "Debit (RK)": "debit_rk",
    "Kredit (RK)": "kredit_rk",
    "Saldo (RK)": "saldo_rk",
    "Debit (BB) - Kredit (RK)": "debit_bb_kredit_rk",
    "Kredit (BB) - Debit (RK)": "kredit_bb_debit_rk",
    "Status": "status",
    "ID": "id",
    "Catatan": "catatan",
}

SUMMARY_KEY_ORDER = ["jenis_saldo","tanggal","buku_besar_bb","rekening_koran_rk","selisih"]
SUMMARY_COL_MAP = {
    "Jenis Saldo": "jenis_saldo",
    "Tanggal": "tanggal",
    "Buku Besar (BB)": "buku_besar_bb",
    "Rekening Koran (RK)": "rekening_koran_rk",
    "Selisih": "selisih",
}

def _df_to_transactions(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    # rename kolom sesuai peta, sisanya di-snake-case agar tetap ikut JSON bila ada kolom tambahan
    rename_map = {c: TRANSACTION_COL_MAP.get(c, _snake(str(c))) for c in df.columns}
    dfr = df.rename(columns=rename_map).copy()

    recs = []
    for _, row in dfr.iterrows():
        obj = {}
        for k in TRANSACTION_KEY_ORDER:
            if k in dfr.columns:
                v = row.get(k)
                if k in ("tanggal_bb","tanggal_rk"):
                    obj[k] = _to_iso_date(v)
                else:
                    # angka tetap angka, "-" jadi None
                    if isinstance(v, str) and v.strip() == "-":
                        obj[k] = None
                    else:
                        # usahakan float/int apa adanya
                        try:
                            if pd.isna(v):
                                obj[k] = None
                            else:
                                obj[k] = float(v) if isinstance(v, (int, float, np.number)) else (str(v) if v not in [None, ""] else None)
                        except Exception:
                            obj[k] = str(v) if v not in [None, ""] else None
        # tambahkan kolom lain yang mungkin ada (di luar order default)
        for k in dfr.columns:
            if k not in obj and k not in TRANSACTION_KEY_ORDER:
                val = row.get(k)
                obj[k] = None if (isinstance(val, str) and val.strip() == "-") else (None if pd.isna(val) else val)
        recs.append(obj)
    return recs

def _df_to_summary(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    dfr = df.rename(columns={c: SUMMARY_COL_MAP.get(c, _snake(str(c))) for c in df.columns}).copy()
    recs = []
    for _, row in dfr.iterrows():
        obj = {}
        for k in SUMMARY_KEY_ORDER:
            if k in dfr.columns:
                v = row.get(k)
                if k == "tanggal":
                    obj[k] = _to_iso_date(v)
                else:
                    if isinstance(v, str) and v.strip() == "-":
                        obj[k] = None
                    else:
                        obj[k] = None if pd.isna(v) else v
        # kolom tambahan lain
        for k in dfr.columns:
            if k not in obj and k not in SUMMARY_KEY_ORDER:
                v = row.get(k)
                obj[k] = None if (isinstance(v, str) and v.strip() == "-") else (None if pd.isna(v) else v)
        recs.append(obj)
    return recs


EASYOCR_READER = None
BANKS_DIR = pathlib.Path("banks")
BANKS_DIR.mkdir(parents=True, exist_ok=True)
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
DYNAMIC_EXTRACTORS: Dict[str, object] = {}
STATIC_EXTRACTORS = {
    "bca": BCAExtractor,
    "bni": BNIExtractor,
    "permata": PermataExtractor,
    "mandiri": MandiriExtractor,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global EASYOCR_READER
    try:
        EASYOCR_READER = easyocr.Reader(["id", "en"])
        load_all_configs_on_startup()
    except Exception as e:
        raise RuntimeError(f"Gagal menginisialisasi OCR: {e}")
    yield
    
app = FastAPI(title="Bank Statement Extraction & Matching API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BankConfigIn(BaseModel):
    name: str = Field(..., description="Kode unik bank", example="BCA, BNI, Permata")
    HEADERS: List[str] = Field(...,
        description="Header kolom transaksi",
        example=["Tanggal", "Keterangan", "Debit", "Kredit", "Saldo"],
    )
    keterangan : str = Field(None, example="Uraian Transaksi")
    kolom_kode : str = Field(None, example="DB/CR")
    target_kode: str = Field(None, example="Amount")
    debit_code : str = Field(None, example="D")
    kredit_code: str = Field(None, example="K")
    header_per_page: bool = Field(True, example=True)
    DATE_FORMAT: str = Field(..., example="dd-MM-yyyy")

    @classmethod
    def as_form(cls,
        name        : str = Form(...),
        HEADERS     : str = Form(...),
        keterangan  : str = Form(None),
        kolom_kode  : str = Form(None),
        target_kode : str = Form(None),
        debit_code  : str = Form(None),
        kredit_code : str = Form(None),
        header_per_page: bool = Form(True),
        DATE_FORMAT : str = Form(...),
    ) -> "BankConfigIn":
        headers_list = [h.strip() for h in (HEADERS or "").split(",") if h.strip()]
        return cls(
            name        = name, HEADERS = headers_list, keterangan  = keterangan,
            kolom_kode  = kolom_kode, target_kode = target_kode, debit_code  = debit_code,
            kredit_code = kredit_code, header_per_page = header_per_page, DATE_FORMAT = DATE_FORMAT,)

    @field_validator("name")
    @classmethod
    def safe_name(cls, v: str) -> str:
        if v is None or not str(v).strip():
            raise ValueError("Field 'nama' wajib diisi")
        if not NAME_RE.match(v):
            raise ValueError("Format nama invalid. Gunakan hanya huruf, angka, underscore atau dash")
        return v.strip()

class BankConfigUpdateForm(BaseModel):
    name        : Optional[str] = None
    HEADERS     : Optional[str] = None
    keterangan  : Optional[str] = None
    kolom_kode  : Optional[str] = None
    target_kode : Optional[str] = None
    debit_code  : Optional[str] = None
    kredit_code : Optional[str] = None
    header_per_page: Optional[bool] = None
    DATE_FORMAT : Optional[str] = None

    @classmethod
    def as_form(cls,
        name        : Optional[str] = Form(None),
        HEADERS     : Optional[str] = Form(None),
        keterangan  : Optional[str] = Form(None),
        kolom_kode  : Optional[str] = Form(None),
        target_kode : Optional[str] = Form(None),
        debit_code  : Optional[str] = Form(None),
        kredit_code : Optional[str] = Form(None),
        header_per_page: Optional[bool] = Form(None),
        DATE_FORMAT : Optional[str] = Form(None),
    ) -> "BankConfigUpdateForm":
        return cls(
            name        = name, HEADERS = HEADERS, keterangan = keterangan,
            kolom_kode  = kolom_kode, target_kode = target_kode, debit_code = debit_code,
            kredit_code = kredit_code, header_per_page = header_per_page, DATE_FORMAT = DATE_FORMAT,)
    @field_validator("name")
    @classmethod
    def safe_name_update(cls, v: str):
        if v is None or not str(v).strip(): 
            raise ValueError("Field 'nama' wajib diisi")
        if not NAME_RE.match(v):
            raise ValueError("Format nama invalid. Gunakan hanya huruf, angka, underscore atau dash")
        return v

### --------- Helper Functions --------- ###
def _path_for(name: str) -> pathlib.Path:
    return BANKS_DIR / f"{name.lower()}.json"

### --------- Dynamic Extractor Registration --------- ###
def make_extractor_class(class_name: str, base_cls, config: dict):
    class DynamicExtractor(base_cls):
        def __init__(self, reader=None):
            try:
                super().__init__(reader)
            except TypeError:
                super(DynamicExtractor, self).__init__(reader)
            for k, v in config.items():
                if isinstance(k, str) and k.isidentifier():
                    setattr(self, k, v)
    DynamicExtractor.__name__ = class_name
    DynamicExtractor.__qualname__ = class_name
    return DynamicExtractor

def register_extractor_from_config(cfg: dict):
    if EASYOCR_READER is None:
        raise RuntimeError("EASYOCR_READER belum diinisialisasi")
    name = cfg.get("name")
    if not name:
        raise ValueError("Konfigurasi harus mengandung 'name'")
    key      = name.lower()
    base_cls = DefaultBaseExtractor
    cls_name = f"{name}Extractor"
    DynamicCls = make_extractor_class(cls_name, base_cls, cfg)
    instance = DynamicCls(EASYOCR_READER)
    DYNAMIC_EXTRACTORS[key] = instance
    return instance

def load_all_configs_on_startup():
    for cfg_file in sorted(BANKS_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            register_extractor_from_config(cfg)
        except Exception as e:
            print(f"Warning: gagal load config {cfg_file}: {e}")

def current_extractors() -> Dict[str, object]:
    static = {k: cls(EASYOCR_READER) for k, cls in STATIC_EXTRACTORS.items()}
    return {**static, **DYNAMIC_EXTRACTORS}

def get_extractor(bank_name: str):
    bank = bank_name.lower()
    extractor = current_extractors().get(bank)
    if not extractor:
        raise HTTPException(status_code=400, detail=f"Bank {bank_name} tidak terdaftar")
    if hasattr(extractor, "reader") and extractor.reader is None:
        if EASYOCR_READER is None:
            raise HTTPException(status_code=500, detail="OCR belum siap")
        extractor.reader = EASYOCR_READER
    return extractor

@app.post("/process")
async def process_file(
    bank_name   : str = Form(...),
    pdf_file    : UploadFile = File(...),
    pages_mode  : str = Form(..., description="All atau custom"),
    pages       : Optional[str] = Form(None, description= "Isi: 1,2,3 jika custom"),
    periode_id  : str = Form(...),
    id_perkiraan: str = Form(...),
    id_department: str = Form(...),
):
    # Validasi pdf
    if not pdf_file.filename.lower().endswith(".pdf") or pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File dokumen bank harus berformat PDF")

    # folder kerja
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("hasil_konversi", exist_ok=True)
    os.makedirs("bukti_matching", exist_ok=True)

    # Simpan PDF sementara
    pdf_path = f"tmp/{pdf_file.filename}"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)

    # Ambil extractor
    extractor = get_extractor(bank_name)

    # Pages arg
    if pages_mode == "all":
        pages_arg = "all"
    else:
        try:
            pages_arg = [int(x.strip()) for x in (pages or "").split(",") if x.strip().isdigit()]
        except Exception:
            raise HTTPException(status_code=400, detail="Format pages tidak valid. Contoh: 1,2,5")

    # 1) Ekstraksi PDF bank → Excel (bank output)
    output_excel_bank = f"hasil_konversi/{bank_name}_output.xlsx"
    try:
        df_bank: pd.DataFrame = extractor.convert(pdf_path, pages=pages_arg, output_excel=output_excel_bank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")
    if df_bank is None or df_bank.empty:
        raise HTTPException(status_code=422, detail="Hasil ekstraksi kosong")

    # 2) Generate Excel Buku Besar dari SQL (pengganti bb_file upload)
    output_excel_bb = f"hasil_konversi/bb_{bank_name}_{periode_id}.xlsx"
    try:
        export_bb_excel_from_sql(periode_id=periode_id,
                                 id_perkiraan=id_perkiraan,
                                 id_department=id_department,
                                 out_path=output_excel_bb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal generate BB dari SQL: {e}")

    # 3) Matching lama (pakai 2 file Excel)
    output_matching = f"bukti_matching/{bank_name}_matching.xlsx"
    try:
        matcher = BankJournalMatcher(
            journal_path=output_excel_bb,      # <-- Excel dari SQL
            bank_path=output_excel_bank,       # <-- Excel dari ekstraksi PDF bank
            output_path=output_matching
        )
        matched_df: pd.DataFrame = matcher.matching()
        summary_df: pd.DataFrame = matcher.df_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proses Matching Gagal: {e}")

    # 4) JSON response (pakai util yang sudah ada di api_query.py)
    transactions_json = _df_to_transactions(matched_df)
    summary_json      = _df_to_summary(summary_df)

    return {
        "status"          : "success",
        "bank"            : bank_name,
        "periode_id"      : periode_id,
        "id_perkiraan"    : id_perkiraan,
        "id_department"   : id_department,
        "transactions"    : transactions_json,
        "summary"         : summary_json,
        "rows_extracted"  : int(len(df_bank)),
        "rows_matched"    : int(len(matched_df)),
        "output_excel_bank": output_excel_bank,
        "output_excel_bb" : output_excel_bb,        # <-- Excel BB hasil SQL
        "output_matching" : output_matching,
    }
