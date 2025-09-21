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

def _save_config(cfg: dict) -> pathlib.Path:
    name = cfg.get("name")
    if not name:
        raise ValueError("Missing name in config")
    p = _path_for(name)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

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

### --------- API Endpoints --------- ###
@app.post("/create-bank")
def create_bank(cfg: BankConfigIn = Depends(BankConfigIn.as_form)):
    key = cfg.name.lower()
    path = _path_for(key)
    if path.exists() or key in STATIC_EXTRACTORS:
        raise HTTPException(status_code=400, detail=f"Bank '{cfg.name}' sudah terdaftar")
    try:
        _save_config(cfg.dict())
        register_extractor_from_config(cfg.dict())
    except Exception as e:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Gagal dalam membuat bank: {e}")
    return JSONResponse(status_code=201, content={"message": f"Bank '{cfg.name}' berhasil dibuat", "bank_key": key})

@app.get("/banks")
def get_bank():
    all_extractors = list(STATIC_EXTRACTORS.keys()) + list(DYNAMIC_EXTRACTORS.keys())
    all_extractors = list(sorted(set(all_extractors)))
    return {"banks": all_extractors}

@app.delete("/bank/{name}")
def delete_bank(name: str):
    key = name.lower()
    p   = _path_for(key)
    if not p.exists() and key not in STATIC_EXTRACTORS:
        raise HTTPException(status_code=404, detail="Bank tidak ditemukan")
    
    try:
        if p.exists():
            p.unlink()
            
        DYNAMIC_EXTRACTORS.pop(key, None)
        STATIC_EXTRACTORS.pop(key, None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus konfigurasi bank: {e}")
    return{"message"     : f"Bank '{name}' dihapus",
        "unregistered": (key not in DYNAMIC_EXTRACTORS and key not in STATIC_EXTRACTORS)}

@app.patch("/bank/{bank_name}/update")
def update_bank(bank_name: str, upd: BankConfigUpdateForm = Depends(BankConfigUpdateForm.as_form)):
    old_key  = bank_name.lower()
    old_path = _path_for(old_key)

    if not old_path.exists():
        raise HTTPException(status_code=404, detail="Bank tidak ditemukan")
    try:
        old_cfg = json.loads(old_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca konfigurasi lama: {e}")

    def is_blank(val):
        if val is None: return True
        if isinstance(val, str) and val.strip() == "": return True
        if isinstance(val, (list, dict)) and len(val) == 0: return True
        return False

    def normalize_headers(val):
        if is_blank(val): return None
        if isinstance(val, str):
            items = [h.strip() for h in val.split(",") if h.strip()]
            if not items: return None
            return items
        raise HTTPException(status_code=422, detail="HEADERS wajib diisi sebagai string 'A,B,C'")

    patch_dict = {}
    if not is_blank(upd.name):
        if not NAME_RE.match(upd.name):
            raise HTTPException(status_code=422, detail="Format nama invalid. Gunakan hanya huruf, angka, underscore atau dash")
        patch_dict["name"] = upd.name
    norm_headers = normalize_headers(upd.HEADERS)
    if norm_headers is not None:
        patch_dict["HEADERS"] = norm_headers

    for k in ["keterangan", "kolom_kode", "target_kode", "debit_code", "kredit_code", "DATE_FORMAT"]:
        v = getattr(upd, k)
        if not is_blank(v):
            patch_dict[k] = v

    if upd.header_per_page is not None:
        patch_dict["header_per_page"] = bool(upd.header_per_page)

    new_name = patch_dict.get("name", old_cfg.get("name") or old_key)
    new_key  = new_name.lower()
    new_path = _path_for(new_key)

    if new_key != old_key and new_path.exists():
        raise HTTPException(status_code=400, detail=f"Bank '{new_name}' sudah terdaftar")

    new_cfg = dict(old_cfg)
    for k, v in patch_dict.items():
        new_cfg[k] = v
    new_cfg["name"] = new_name

    try:
        _save_config(new_cfg)
        if new_key != old_key:
            try:
                old_path.unlink(missing_ok=True)
            except Exception as e_del:
                try:
                    new_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail=f"Gagal menghapus file lama: {e_del}")

        DYNAMIC_EXTRACTORS.pop(old_key, None)
        try:
            register_extractor_from_config(new_cfg)
        except Exception as e_reg:
            try:
                if new_key != old_key:
                    try:
                        new_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    old_path.write_text(json.dumps(old_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    old_path.write_text(json.dumps(old_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            try:
                register_extractor_from_config(old_cfg)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Gagal melakukan register extractor: {e_reg}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui konfigurasi: {e}")

    return JSONResponse(
        status_code=200,
        content={"message": f"Bank '{old_key}' diperbarui (form)",
            "bank_key": new_key, "changed_fields": list(patch_dict.keys()),
            "before": old_cfg, "after": new_cfg,},)

@app.post("/process")
async def process_file(
    bank_name : str = Form(...),
    pdf_file  : UploadFile = File(...),
    bb_file   : UploadFile = File(...),
    pages_mode: str = Form(..., description="All atau custom"),
    pages     : Optional[str] = Form(None, description= "Isi: 1, 2, 3 jika custom"),
):
    if not pdf_file.filename.lower().endswith(".pdf") or pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File dokumen bank harus berformat PDF")
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("hasil_konversi", exist_ok=True)
    os.makedirs("bukti_matching", exist_ok=True)
    pdf_path = f"tmp/{pdf_file.filename}"
    bb_path = f"tmp/{bb_file.filename}"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)
    with open(bb_path, "wb") as buffer:
        shutil.copyfileobj(bb_file.file, buffer)
    extractor = get_extractor(bank_name)
    if pages_mode == "all":
        pages_arg = "all"
    else:
        try:
            pages_arg = [int(x.strip()) for x in (pages or "").split(",") if x.strip().isdigit()]
        except Exception:
            raise HTTPException(status_code=400, detail="Format pages tidak valid. Gunakan contoh: 1, 2, 5")
    output_excel = f"hasil_konversi/{bank_name}_output.xlsx"
    try:
        df_bank: pd.DataFrame = extractor.convert(pdf_path, pages=pages_arg, output_excel=output_excel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")
    if df_bank is None or df_bank.empty:
        raise HTTPException(status_code=422, detail="Hasil ekstraksi kosong")
    output_matching = f"bukti_matching/{bank_name}_matching.xlsx"
    try:
        matcher = BankJournalMatcher(journal_path=bb_path, bank_path=output_excel, output_path=output_matching)
        matched_df: pd.DataFrame = matcher.matching()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proses Matching Gagal: {e}")
    return {
        "status"         : "success",
        "bank"           : bank_name,
        "rows_extracted" : int(len(df_bank)),
        "rows_matched"   : int(len(matched_df)),
        "output_excel"   : output_excel,
        "output_matching": output_matching,
    }

@app.get("/download/{bank_name}")
async def download_result(bank_name: str):
    filepath = f"bukti_matching/{bank_name}_matching.xlsx"
    if os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=os.path.basename(filepath),
        )
    return JSONResponse({"error": "File not found"}, status_code=404)