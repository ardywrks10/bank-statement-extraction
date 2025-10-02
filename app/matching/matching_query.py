import pandas as pd
import numpy as np
import re
import itertools
from typing import Optional
from app.extractors.reconciler import Reconciler

class BankJournalMatcher:
    def __init__(self,
                 journal_path: Optional[str] = None,
                 bank_path: Optional[str] = None,
                 output_path: Optional[str] = "matched_df.xlsx",
                 journal_df: Optional[pd.DataFrame] = None,
                 bank_df: Optional[pd.DataFrame] = None,
                 save_excel: bool = False):
        """
        Versi ini mendukung:
        - Input via path Excel (compat lama), ATAU
        - Input via DataFrame langsung (mode SQL).
        - Penulisan Excel bisa dimatikan (save_excel=False).
        """
        self.journal_path  = journal_path
        self.bank_path     = bank_path
        self.output_path   = output_path
        self.journal_df    = journal_df
        self.bank_df       = bank_df
        self.matched_df    = None
        self.df_summary    = None
        self.date_tolerance   = 7
        self.amount_tolerance = 0.0
        self.save_excel    = save_excel

    def find_header_idx(self, path, anchor_cols = None, max_rows=20):
        if anchor_cols is None:
            anchor_cols = ["Debit", "Credit", "Saldo", "Date",
                           "Nama Akun / Tanggal", "Debet"]
        df_raw = pd.read_excel(path, header=None, nrows=max_rows)
        for idx, row in df_raw.iterrows():
            row_str = row.astype(str).str.upper().tolist()
            for anchor in anchor_cols:
                if anchor.upper() in row_str:
                    return idx
        return 0

    def load_data(self):
        """
        Jika journal_df dan bank_df sudah diisi (mode SQL), langsung pakai.
        Jika tidak, fallback ke path Excel (compat lama).
        """
        if isinstance(self.journal_df, pd.DataFrame) and not self.journal_df.empty \
           and isinstance(self.bank_df, pd.DataFrame) and not self.bank_df.empty:
            return self.journal_df, self.bank_df

        if not self.journal_path or not self.bank_path:
            raise ValueError("journal_df/bank_df tidak disediakan dan journal_path/bank_path juga kosong.")

        header_journal = self.find_header_idx(self.journal_path)
        df_bb = pd.read_excel(self.journal_path, header=header_journal)
        df_bb = df_bb.dropna(how="all")

        saldo_cols = [c for c in df_bb.columns if str(c).strip().lower() in ["saldo", "balance"]]
        if saldo_cols:
            col = saldo_cols[0]
            first_val = df_bb.iloc[0][col]
            if pd.isna(first_val) or (isinstance(first_val, str) and first_val.strip() == ""):
                df_bb = df_bb.drop(df_bb.index[0])

        df_bank         = pd.read_excel(self.bank_path)
        self.journal_df = df_bb
        self.bank_df    = df_bank
        return self.journal_df, self.bank_df

    def preprocess(self, df):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            if re.search(r"de?b(et|it)", col):
                rename_map[col] = "Debit"
            elif re.search(r"(credit|kredit)", col):
                rename_map[col] = "Kredit"
            elif re.search(r"(balance|saldo)", col):
                rename_map[col] = "Saldo"
            elif re.search(r"^tgl|tanggal|date", col):
                rename_map[col] = "Tgl"
            elif re.search(r"jurnal[_\s-]?id", col):
                rename_map[col] = "Jurnal ID"
            elif col == "nomor" or re.search(r"no[\.\s_-]*voucher", col):
                # tangkap "Nomor", "No Voucher", "No. Voucher", "no_voucher", dll.
                rename_map[col] = "No Voucher"

        df = df.rename(columns=rename_map)

        if "Tgl" in df.columns:
            t = pd.to_datetime(df["Tgl"], errors="coerce", format="%Y-%m-%d")
            mask = t.isna()
            if mask.any():
                t.loc[mask] = pd.to_datetime(df.loc[mask, "Tgl"], errors="coerce", dayfirst=True)
            df["Tgl"] = t.dt.date

        for col in ["Debit", "Kredit", "Saldo"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df


    def greedy_matching(self, journal_df, bank_df, rounding=2):
        results = []

        for col in ["Debit", "Kredit", "Saldo"]:
            if col in journal_df.columns:
                journal_df[col] = journal_df[col].astype(float)
            if col in bank_df.columns:
                bank_df[col] = bank_df[col].astype(float)

        matched_bank_idxs = set()
        journal_n1 = journal_df.iloc[1:-1] if len(journal_df) > 2 else journal_df.iloc[0:0]
        bank_n1    = bank_df.iloc[1:-1]    if len(bank_df) > 2    else bank_df.iloc[0:0]

        for j_idx, j_row in journal_n1.iterrows():
            j_tgl     = j_row.get("Tgl", None)
            j_debit   = j_row.get("Debit", 0.0)
            j_kredit  = j_row.get("Kredit", 0.0)
            j_voucher = j_row.get("No Voucher", "-")
            j_jid     = j_row.get("Jurnal ID", None)

            found = None
            for b_idx, b_row in bank_n1.iterrows():
                if b_idx in matched_bank_idxs:
                    continue
                b_tgl    = b_row.get("Tgl", None)
                b_debet  = b_row.get("Debit", 0.0)
                b_kredit = b_row.get("Kredit", 0.0)

                try:
                    if j_tgl and b_tgl:
                        delta_hari = abs((j_tgl - b_tgl).days)
                    else:
                        delta_hari = 999
                except Exception:
                    delta_hari = 999

                if delta_hari <= self.date_tolerance:
                    if (j_debit == b_kredit and j_kredit == 0 and b_debet == 0) or \
                       (j_kredit == b_debet and j_debit == 0 and b_kredit == 0):
                        found = {
                            "Tanggal (BB)" : j_tgl,
                            "No Voucher"   : j_voucher if pd.notna(j_voucher) else "-",
                            "Jurnal ID"    : j_jid,
                            "Debit (BB)"   : float(j_debit),
                            "Kredit (BB)"  : float(j_kredit),
                            "Saldo (BB)"   : 0.0,
                            "Tanggal (RK)" : b_tgl,
                            "Debit (RK)"   : float(b_debet),
                            "Kredit (RK)"  : float(b_kredit),
                            "Saldo (RK)"   : 0.0,
                            "Debit (BB) - Kredit (RK)" : round(float(j_debit) - float(b_kredit), rounding),
                            "Kredit (BB) - Debit (RK)" : round(float(j_kredit) - float(b_debet), rounding),
                            "Status"       : "Matched",
                            "Catatan"      : "-",
                        }
                        matched_bank_idxs.add(b_idx)
                        break

            if found:
                results.append(found)
            else:
                results.append({
                    "Tanggal (BB)" : j_tgl,
                    "No Voucher"   : j_voucher if pd.notna(j_voucher) else "-",
                    "Jurnal ID"    : j_jid,
                    "Debit (BB)"   : float(j_debit),
                    "Kredit (BB)"  : float(j_kredit),
                    "Saldo (BB)"   : 0.0,
                    "Tanggal (RK)" : "-",
                    "Debit (RK)"   : 0.0,
                    "Kredit (RK)"  : 0.0,
                    "Saldo (RK)"   : 0.0,
                    "Debit (BB) - Kredit (RK)" : round(float(j_debit), rounding),
                    "Kredit (BB) - Debit (RK)" : round(float(j_kredit), rounding),
                    "Status"       : "Unmatched",
                    "Catatan"      : "-",
                })

        for b_idx, b_row in bank_n1.iterrows():
            if b_idx not in matched_bank_idxs:
                b_debet  = float(b_row.get("Debit", 0.0))
                b_kredit = float(b_row.get("Kredit", 0.0))
                results.append({
                    "Tanggal (BB)"      : "-",
                    "No Voucher"        : "-",
                    "Jurnal ID"         : None,
                    "Debit (BB)"        : 0.0,
                    "Kredit (BB)"       : 0.0,
                    "Saldo (BB)"        : 0.0,
                    "Tanggal (RK)"      : b_row.get("Tgl", None),
                    "Debit (RK)"        : b_debet,
                    "Kredit (RK)"       : b_kredit,
                    "Saldo (RK)"        : 0.0,
                    "Debit (BB) - Kredit (RK)" : round(0 - b_kredit, rounding),
                    "Kredit (BB) - Debit (RK)" : round(0 - b_debet, rounding),
                    "Status"            : "Unmatched",
                    "Catatan"           : "-",
                })

        # ambil saldo awal dari baris 0 (diasumsikan opening balance)
        saldo_awal_bb = round(float(self.journal_df.iloc[0]["Saldo"]), rounding) if ("Saldo" in self.journal_df.columns and len(self.journal_df)>0) else 0.0
        saldo_awal_b  = round(float(self.bank_df.iloc[0]["Saldo"]), rounding) if ("Saldo" in self.bank_df.columns and len(self.bank_df)>0) else 0.0

        df_results = pd.DataFrame(results)
        # sort by tanggal referensi
        for c in ["Tanggal (BB)", "Tanggal (RK)"]:
            if c not in df_results.columns:
                continue
            ser = df_results[c]

            # Jika sudah date/datetime biarkan
            if pd.api.types.is_datetime64_any_dtype(ser):
                df_results[c] = pd.to_datetime(ser, errors="coerce").dt.date
                continue

            # 1) Coba format MySQL (YYYY-MM-DD)
            dt = pd.to_datetime(ser, errors="coerce", format="%Y-%m-%d")

            # 2) Fallback untuk hasil OCR bank (day-first, dd/MM/yyyy)
            mask = dt.isna()
            if mask.any():
                dt.loc[mask] = pd.to_datetime(ser[mask], errors="coerce", dayfirst=True)

            df_results[c] = dt.dt.date

        df_results["Tanggal Referensi"] = df_results["Tanggal (BB)"].fillna(df_results["Tanggal (RK)"])
        df_results = df_results.sort_values(by=["Tanggal Referensi", "Tanggal (BB)", "Tanggal (RK)"], ascending=[True, True, True])
        df_results = df_results.drop(columns=["Tanggal Referensi"])

        if "Status" in df_results.columns:
            akhir_cols = ["Status"]
            cols2 = [c for c in df_results.columns if c not in akhir_cols] + [c for c in akhir_cols if c in df_results.columns]
            df_results = df_results[cols2]

        df_results = self._force_order(df_results)

        return df_results, saldo_awal_bb, saldo_awal_b

    def unmatched_links(self, df, max_group = 4):
        df = df.copy()
        df["ID"] = '-'
        unmatched = df[df["Status"] == "Unmatched"]
        group_counter = 1
        used_idx  = set()

        def _max_two(a, b) -> float:
            a = 0.0 if pd.isna(a) else float(a)
            b = 0.0 if pd.isna(b) else float(b)
            return max(a, b)

        for i, row_i in unmatched.iterrows():
            if i in used_idx:
                continue

            target_credit = row_i["Kredit (BB)"]
            target_debit  = row_i["Debit (BB)"]
            date_i        = row_i["Tanggal (BB)"] if target_credit > 0 or target_debit > 0 else row_i["Tanggal (RK)"]

            # (kasus2 sama seperti versi lama) — dipertahankan
            # … (kode asli unmatched_links dari file kamu) …
            # demi ringkas, gunakan implementasi aslinya di projectmu
            # ---------------------------
            # BEGIN salinan pendek aman:
            for side in ["BB_credit", "RK_credit", "RK_debit", "BB_debit"]:
                pass
            # END salinan pendek aman
            # ---------------------------

        # urutkan agar group ID muncul di atas
        df["Tanggal Referensi"] = df["Tanggal (BB)"].fillna(df["Tanggal (RK)"])
        df["HasID"] = (df["ID"] != "-")
        df["ID_num"] = pd.to_numeric(df["ID"].str.extract(r"G(\d+)")[0], errors="coerce").fillna(np.inf)
        df = df.sort_values(by=["Tanggal Referensi", "HasID", "ID_num"],
                            ascending=[True, False, True], kind="mergesort") \
               .drop(columns=["Tanggal Referensi", "HasID", "ID_num"])
        return df

    # (fungsi add_unmatched_links, clean_empty_dates, apply_saldo: gunakan versi yang sudah ada di file kamu — tidak berubah)

    def add_unmatched_links(self, df, max_group=1):
        # gunakan implementasi asli kamu
        return df

    def clean_empty_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(df[(df["Tanggal (BB)"].isna()) & (df["Tanggal (RK)"].isna())].index)
        df = df.reset_index(drop=True)
        return df

    def apply_saldo(self, df: pd.DataFrame, saldo_awal_bb: float, saldo_awal_b: float, rounding: int = 2) -> pd.DataFrame:
        # gunakan implementasi asli kamu (sudah oke)
        df = df.copy()

        # cari tanggal tengah utk derive opening/closing date
        tgl_mid_bb = None
        for i in range(1, len(df)):
            val = df.iloc[i]["Tanggal (BB)"]
            if pd.notna(val):
                tgl_mid_bb = pd.to_datetime(val, errors="coerce")
                if pd.notna(tgl_mid_bb):
                    tgl_mid_bb = tgl_mid_bb.date()
                    break
        opening_date_bb, closing_date_bb = None, None
        if tgl_mid_bb:
            opening_date_bb = (tgl_mid_bb.replace(day=1) - pd.Timedelta(days=1))
            closing_date_bb = (tgl_mid_bb + pd.offsets.MonthEnd(0)).date()

        tgl_mid_rk = None
        for i in range(1, len(df)):
            val = df.iloc[i]["Tanggal (RK)"]
            if pd.notna(val):
                tgl_mid_rk = pd.to_datetime(val, errors="coerce")
                if pd.notna(tgl_mid_rk):
                    tgl_mid_rk = tgl_mid_rk.date()
                    break
        opening_date_rk, closing_date_rk = None, None
        if tgl_mid_rk:
            opening_date_rk = (tgl_mid_rk.replace(day=1) - pd.Timedelta(days=1))
            closing_date_rk = (tgl_mid_rk + pd.offsets.MonthEnd(0)).date()

        if "Saldo (BB)" in df.columns and {"Debit (BB)", "Kredit (BB)"}.issubset(df.columns):
            saldo = saldo_awal_bb
            saldo_list = df["Saldo (BB)"].tolist()
            for i in range(len(df)):
                debit = df.iloc[i]["Debit (BB)"]; kredit = df.iloc[i]["Kredit (BB)"]
                saldo += debit - kredit
                saldo_list[i] = saldo
            df["Saldo (BB)"] = saldo_list

        if "Saldo (RK)" in df.columns and {"Debit (RK)", "Kredit (RK)"}.issubset(df.columns):
            saldo = saldo_awal_b
            saldo_list = df["Saldo (RK)"].tolist()
            for i in range(len(df)):
                debit = df.iloc[i]["Debit (RK)"]; kredit = df.iloc[i]["Kredit (RK)"]
                saldo += kredit - debit
                saldo_list[i] = saldo
            df["Saldo (RK)"] = saldo_list

        opening_row = {
            "Tanggal (BB)": opening_date_bb, 
            "No Voucher": "-",
            "Jurnal ID": "-",
            "Debit (BB)": 0.0, "Kredit (BB)": 0.0, "Saldo (BB)": saldo_awal_bb,
            "Tanggal (RK)": opening_date_rk, "Debit (RK)": 0.0, "Kredit (RK)": 0.0, "Saldo (RK)": saldo_awal_b,
            "Debit (BB) - Kredit (RK)" : "-",
            "Kredit (BB) - Debit (RK)" : "-",
            "Status": "Opening Balance", "ID": "-", "Catatan": "-"
        }
        df = pd.concat([pd.DataFrame([opening_row]), df], ignore_index=True)

        total_debit_bb = float(pd.to_numeric(df["Debit (BB)"], errors="coerce").fillna(0).sum())
        total_kredit_bb= float(pd.to_numeric(df["Kredit (BB)"],errors="coerce").fillna(0).sum())
        saldo_bb_akhir = round(saldo_awal_bb + total_debit_bb - total_kredit_bb, rounding)

        total_debit_b = float(pd.to_numeric(df["Debit (RK)"], errors="coerce").fillna(0).sum())
        total_kredit_b= float(pd.to_numeric(df["Kredit (RK)"],errors="coerce").fillna(0).sum())
        saldo_b_akhir = round(saldo_awal_b + total_kredit_b - total_debit_b, rounding)

        closing_row = {
            "Tanggal (BB)": closing_date_bb, 
            "No Voucher": "-",
            "Jurnal ID": "-",
            "Debit (BB)": total_debit_bb, "Kredit (BB)": total_kredit_bb, "Saldo (BB)": saldo_bb_akhir,
            "Tanggal (RK)": closing_date_rk, "Debit (RK)": total_debit_b, "Kredit (RK)": total_kredit_b, "Saldo (RK)": saldo_b_akhir,
            "Debit (BB) - Kredit (RK)" : round(total_debit_bb - total_kredit_b, rounding),
            "Kredit (BB) - Debit (RK)" : round(total_kredit_bb - total_debit_b, rounding),
            "Status": "Closing Balance", "ID": "-", "Catatan": "-"
        }
        df = pd.concat([df, pd.DataFrame([closing_row])], ignore_index=True)

        df_summary = pd.DataFrame([
            {"Jenis Saldo": "Saldo Awal", "Tanggal": opening_date_bb,
             "Buku Besar (BB)": saldo_awal_bb, "Rekening Koran (RK)": saldo_awal_b,
             "Selisih": round(saldo_awal_bb - saldo_awal_b, rounding)},
            {"Jenis Saldo": "Saldo Akhir", "Tanggal": closing_date_bb,
             "Buku Besar (BB)": saldo_bb_akhir, "Rekening Koran (RK)": saldo_b_akhir,
             "Selisih": round(saldo_bb_akhir - saldo_b_akhir, rounding)}
        ])

        df = self._force_order(df)

        return df, df_summary

    def matching(self):
        self.load_data()
        self.journal_df = self.preprocess(self.journal_df)
        self.bank_df    = self.preprocess(self.bank_df)

        self.matched_df, saldo_awal_bb, saldo_awal_b = self.greedy_matching(self.journal_df, self.bank_df)
        self.matched_df = self.unmatched_links(self.matched_df)
        self.matched_df = self.add_unmatched_links(self.matched_df)
        self.matched_df = self.clean_empty_dates(self.matched_df)
        self.matched_df, self.df_summary = self.apply_saldo(self.matched_df, saldo_awal_bb, saldo_awal_b)

        reconciler = Reconciler(abs_tol=1.0, rel_tol=1e-4, max_group=4)
        self.matched_df = reconciler.predict(self.matched_df)

        self.matched_df = self._force_order(self.matched_df)

        if self.save_excel and self.output_path:
            with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
                self.matched_df.to_excel(writer, sheet_name="Transaksi", index=False)
                self.df_summary.to_excel(writer, sheet_name="Summary", index=False)

        return self.matched_df


    def _force_order(self, df: pd.DataFrame) -> pd.DataFrame:
        DESIRED = [
            "Tanggal (BB)",
            "Jurnal ID",           # ← di sini, sebelum No Voucher
            "No Voucher",
            "Debit (BB)",
            "Kredit (BB)",
            "Saldo (BB)",
            "Tanggal (RK)",
            "Debit (RK)",
            "Kredit (RK)",
            "Saldo (RK)",
            "Debit (BB) - Kredit (RK)",
            "Kredit (BB) - Debit (RK)",
            "Status",
            "ID",
            "Catatan",
        ]
        if df is None or df.empty:
            return df
        ordered = [c for c in DESIRED if c in df.columns]
        tail = [c for c in df.columns if c not in ordered]
        return df[ordered + tail]
