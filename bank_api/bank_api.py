# ---------------------------------------
# --------- Importing Packages ----------
# ---------------------------------------

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import pandas as pd
import easyocr

# ---------------------------------------
# ---- Importing Extractor & Matcher ----
# ---------------------------------------

from bca import BCAExtractor
from bni import BNIExtractor
from permata import PermataExtractor
from mandiri import MandiriExtractor
from matching import BankJournalMatcher

app = FastAPI(title="Bank Statement Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

EASYOCR_READER = easyocr.Reader(['id', 'en'])
EXTRACTORS = {
    "bca"     : BCAExtractor(EASYOCR_READER),
    "bni"     : BNIExtractor(EASYOCR_READER),
    "permata" : PermataExtractor(EASYOCR_READER),
    "mandiri" : MandiriExtractor(EASYOCR_READER),
}

def get_extractor(bank_name: str):
    bank = bank_name.lower()
    if bank not in EXTRACTORS:
        raise HTTPException(status_code=400, detail=f"Bank {bank_name} tidak terdaftar")
    return EXTRACTORS[bank]

@app.post("/process")
async def process_file(
    bank_name  : str = Form(...),
    pdf_file   : UploadFile = File(...),
    bb_file    : UploadFile = File(...),
    pages_mode : str = Form("all"),
    pages      : str = Form(None)
):
    if not pdf_file.filename.lower().endswith(".pdf") or pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File dokumen bank harus berformat PDF")
    
    # --- 1. Temporary Folder ---
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("hasil_konversi", exist_ok=True)
    os.makedirs("bukti_matching", exist_ok=True)
    
    pdf_path = f"tmp/{pdf_file.filename}"
    bb_path  = f"tmp/{bb_file.filename}"
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)
    with open(bb_path, "wb") as buffer:
        shutil.copyfileobj(bb_file.file, buffer)    
    
    # --- 2. Memilih Extractor --- 
    extractor = get_extractor(bank_name)
        
    # --- 3. Menentukan pages sesuai mode ---
    if pages_mode == "all":
        pages_arg = "all"
    else:
        try:
            pages_arg = [int(x.strip()) for x in pages.split(",") if x.strip().isdigit()]
        except Exception:
            raise HTTPException(status_code=400, detail="Format pages tidak valid. Gunakan contoh: 1, 2, 5")
        
    # --- 4. Extraksi PDF ke format Excel ---
    output_excel = f"hasil_konversi/{bank_name}_output.xlsx"
    try:
        df_bank: pd.DataFrame = extractor.convert(pdf_path, pages=pages_arg, output_excel=output_excel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")
    
    if df_bank is None or df_bank.empty:
        raise HTTPException(status_code=422, detail="Hasil ekstraksi kosong")
    
    # --- 5. Matching Buku Besar (BB) dan Rekening ---
    output_matching = f"bukti_matching/{bank_name}_matching.xlsx"
    try:
        matcher = BankJournalMatcher(
            journal_path = bb_path,
            bank_path    = output_excel,
            output_path  = output_matching
        )
        
        matched_df: pd.DataFrame = matcher.matching()
    except Exception as e:
        raise HTTPException(status_code=500, detail= f"Proses Matching Gagal: {e}")
    
    return {
        "status": "success",
        "bank"  : bank_name,
        "rows_extracted"  : len(df_bank),
        "rows_matched"    : len(matched_df),
        "output_excel"    : output_excel,
        "output_matching" : output_matching
    }

@app.get("/download/{bank_name}")
async def download_result(bank_name: str):
    filepath = f"bukti_matching/{bank_name}_matching.xlsx"
    if os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=os.path.basename(filepath)
        )
    return JSONResponse({"error": "File not found"}, status_code=404)
