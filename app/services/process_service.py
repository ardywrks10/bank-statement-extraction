import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
import pandas as pd

from app.utils import _df_to_transactions, _df_to_summary
from app.extractors.dynamic_registry import get_extractor               # ⬅️ pakai registry tunggal
from app.sql.sql_bb_export import export_bb_excel_from_sql
from app.matching.matching_query import BankJournalMatcher

from app.paths import TMP_DIR, KONVERSI_DIR, MATCHING_DIR

for d in [TMP_DIR, KONVERSI_DIR, MATCHING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class ProcessService:
    def run(
        self,
        bank_name: str,
        pdf_file: UploadFile,
        pages_mode: str,
        pages: str | None,
        periode_id: str,
        id_perkiraan: str,
        id_department: str,
    ):
        # 1) Simpan PDF
        pdf_path = TMP_DIR / pdf_file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)

        # 2) Extractor
        extractor = get_extractor(bank_name)

        # 3) Pages arg
        if pages_mode == "all":
            pages_arg = "all"
        else:
            try:
                pages_arg = [int(x.strip()) for x in (pages or "").split(",") if x.strip().isdigit()]
            except Exception:
                raise HTTPException(status_code=400, detail="Format pages tidak valid. Contoh: 1,2,5")

        # 4) Ekstraksi bank → Excel
        output_excel_bank = KONVERSI_DIR / f"{bank_name}_output_{periode_id}.xlsx"
        try:
            df_bank: pd.DataFrame = extractor.convert(str(pdf_path), pages=pages_arg, output_excel=str(output_excel_bank))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ekstraksi gagal: {e}")

        if df_bank is None or df_bank.empty:
            raise HTTPException(status_code=422, detail="Hasil ekstraksi kosong")

        # 5) Export BB dari SQL
        output_excel_bb = KONVERSI_DIR / f"bb_{bank_name}_{periode_id}.xlsx"
        try:
            export_bb_excel_from_sql(
                periode_id=periode_id,
                id_perkiraan=id_perkiraan,
                id_department=id_department,
                out_path=str(output_excel_bb),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal generate BB: {e}")

        # 6) Matching
        output_matching = MATCHING_DIR / f"{bank_name}_matching_{periode_id}.xlsx"
        try:
            matcher = BankJournalMatcher(
                journal_path=str(output_excel_bb),
                bank_path=str(output_excel_bank),
                output_path=str(output_matching),
                save_excel=True,
            )
            matched_df: pd.DataFrame = matcher.matching()
            summary_df: pd.DataFrame = matcher.df_summary
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proses matching gagal: {e}")

        # 7) Mapping → JSON
        transactions_json = _df_to_transactions(matched_df)
        summary_json = _df_to_summary(summary_df)

        return {
            "status": "success",
            "bank": bank_name,
            "periode_id": periode_id,
            "id_perkiraan": id_perkiraan,
            "id_department": id_department,
            "transactions": transactions_json,
            "summary": summary_json,
            "rows_extracted": int(len(df_bank)),
            "rows_matched": int(len(matched_df)),
            "output_excel_bank": str(output_excel_bank),
            "output_excel_bb": str(output_excel_bb),
            "output_matching": str(output_matching),
        }
    
    # Re-Match
    def rematch(
        self,
        bank_name: str,
        periode_id: str,
        id_perkiraan: str,
        id_department: str,
        rk_filename: str | None = None,
        save_excel: bool = True,
    ):
        """
        Re-run matching TANPA OCR:
        - Ambil RK dari Excel tersimpan (hasil ekstraksi sebelumnya).
        - Re-query BB dari SQL (TANPA caching).
        - Matching ulang → simpan matching excel ber-periode.
        """

        # 1) Tentukan path RK tersimpan
        if rk_filename:
            rk_path = KONVERSI_DIR / rk_filename
        else:
            rk_path = KONVERSI_DIR / f"{bank_name}_output_{periode_id}.xlsx"

        if not rk_path.exists():
            raise HTTPException(status_code=404, detail=f"RK Excel tidak ditemukan: {rk_path.name}")

        try:
            df_bank = pd.read_excel(rk_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal membaca RK Excel: {e}")

        if df_bank is None or df_bank.empty:
            raise HTTPException(status_code=422, detail="RK Excel kosong")

        # 2) Re-generate BB dari SQL (tanpa caching)
        output_excel_bb = KONVERSI_DIR / f"bb_{bank_name}_{periode_id}.xlsx"
        try:
            export_bb_excel_from_sql(
                periode_id=periode_id,
                id_perkiraan=id_perkiraan,
                id_department=id_department,
                out_path=str(output_excel_bb),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal generate BB: {e}")

        # 3) Matching ulang (gunakan PATH keduanya agar sheet & penamaan konsisten)
        output_matching = MATCHING_DIR / f"{bank_name}_matching_{periode_id}.xlsx"
        try:
            matcher = BankJournalMatcher(
                journal_path=str(output_excel_bb),
                bank_path=str(rk_path),
                output_path=str(output_matching),
                save_excel=save_excel,
            )
            matched_df: pd.DataFrame = matcher.matching()
            summary_df: pd.DataFrame = matcher.df_summary
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proses re-matching gagal: {e}")

        # 4) Mapping JSON
        from app.utils import _df_to_transactions, _df_to_summary
        transactions_json = _df_to_transactions(matched_df)
        summary_json = _df_to_summary(summary_df)

        return {
            "status": "success",
            "bank": bank_name,
            "periode_id": periode_id,
            "id_perkiraan": id_perkiraan,
            "id_department": id_department,
            "transactions": transactions_json,
            "summary": summary_json,
            "rows_rk": int(len(df_bank)),
            "rows_bb": int(len(pd.read_excel(output_excel_bb))),
            "rows_matched": int(len(matched_df)),
            "output_rk": str(rk_path),
            "output_bb": str(output_excel_bb),
            "output_matching": str(output_matching),
        }

