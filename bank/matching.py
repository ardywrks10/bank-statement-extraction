import pandas as pd
import numpy as np
import re
import itertools
from datetime import timedelta

class BankJournalMatcher:
    def __init__(self, journal_path, bank_path, 
                 output_path = "matched_df.xlsx"):
        self.journal_path  = journal_path
        self.bank_path     = bank_path
        self.output_path   = output_path
        self.journal_df    = None
        self.bank_df       = None
        self.matched_df    = None
        self.df_summary    = None
        self.date_tolerance = 7
        self.amount_tolerance = 0.0

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
            elif re.search(r"no[-_\s]?voucher", col):   
                rename_map[col] = "No Voucher"
        df = df.rename(columns=rename_map)

        date_pattern = re.compile(r"(tgl|tanggal|date)", re.IGNORECASE)
        for col in df.columns:
            if date_pattern.search(col):
                df = df.rename(columns={col: "Tgl"})
                break
        if "Tgl" in df.columns:
            df["Tgl"] = pd.to_datetime(df["Tgl"], errors="coerce", dayfirst=True).dt.date

        numeric_cols = ["Debit", "Kredit", "Saldo"]
        for col in numeric_cols:
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
        journal_n1 = journal_df.iloc[1:-1]  
        bank_n1    = bank_df.iloc[1:-1]     
    
        for j_idx, j_row in journal_n1.iterrows():
            j_tgl     = j_row["Tgl"]
            j_debit   = j_row["Debit"]
            j_kredit  = j_row["Kredit"]
            j_voucher = j_row["No Voucher"] 

            found = None
            for b_idx, b_row in bank_n1.iterrows():
                if b_idx in matched_bank_idxs:
                    continue
                b_tgl    = b_row["Tgl"]
                b_debet  = b_row["Debit"]
                b_kredit = b_row["Kredit"]

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
                            "Debit (BB)"   : j_debit,
                            "Kredit (BB)"  : j_kredit,
                            "Saldo (BB)"   : 0.0,
                            "Tanggal (RK)" : b_tgl,
                            "Debit (RK)"   : b_debet,
                            "Kredit (RK)"  : b_kredit,
                            "Saldo (RK)"   : 0.0,
                            "Debit (BB) - Kredit (RK)" : round(j_debit - b_kredit, rounding),
                            "Kredit (BB) - Debit (RK)" : round(j_kredit - b_debet, rounding),
                            "Status"       : "Matched",
                            "Catatan"      : "-"
                        }
                        matched_bank_idxs.add(b_idx)
                        break
            if found:
                results.append(found)
            else:
                results.append({
                    "Tanggal (BB)" : j_tgl,
                    "No Voucher"   : j_voucher if pd.notna(j_voucher) else "-",
                    "Debit (BB)"   : j_debit,
                    "Kredit (BB)"  : j_kredit,
                    "Saldo (BB)"   : 0.0,
                    "Tanggal (RK)" : "-",
                    "Debit (RK)"   : 0.0,
                    "Kredit (RK)"  : 0.0,
                    "Saldo (RK)"   : 0.0,
                    "Debit (BB) - Kredit (RK)" : round(j_debit, rounding),
                    "Kredit (BB) - Debit (RK)" : round(j_kredit, rounding),
                    "Status"       : "Unmatched",
                    "Catatan"      : "-"
                })

        for b_idx, b_row in bank_n1.iterrows():
            if b_idx not in matched_bank_idxs:
                b_debet  = b_row["Debit"]
                b_kredit = b_row["Kredit"]
                results.append({
                    "Tanggal (BB)"      : "-",
                    "No Voucher"        : "-",
                    "Debit (BB)"        : 0.0,
                    "Kredit (BB)"       : 0.0,
                    "Saldo (BB)"        : 0.0,
                    "Tanggal (RK)"      : b_row["Tgl"],
                    "Debit (RK)"        : b_debet,
                    "Kredit (RK)"       : b_kredit,
                    "Saldo (RK)"        : 0.0,
                    "Debit (BB) - Kredit (RK)" : round(0-b_kredit, rounding),
                    "Kredit (BB) - Debit (RK)" : round(0-b_debet, rounding),
                    "Status"            : "Unmatched",
                    "Catatan"           : "-"
                })
                
        saldo_awal   = round(float(journal_df.iloc[0]["Saldo"]), rounding)
        saldo_awal_b = round(float(bank_df.iloc[0]["Saldo"]), rounding)
        df_results = pd.DataFrame(results)
        cols       = ["Tanggal (BB)", "Tanggal (RK)"]
        df_results[cols]  = df_results[cols].apply(
            lambda col: pd.to_datetime(col, format="%d/%m/%Y", errors="coerce").dt.date)
        df_results["Tanggal Referensi"] = df_results["Tanggal (BB)"].fillna(df_results["Tanggal (RK)"])
        df_results = df_results.sort_values(by=["Tanggal Referensi", "Tanggal (BB)", 
                                                "Tanggal (RK)"], ascending=[True, True, True])
        df_results = df_results.drop(columns=["Tanggal Referensi"])
        if "Status" in df_results.columns:
            akhir_cols = ["Status"]
            cols = [c for c in df_results.columns if c not in akhir_cols] \
                 + [c for c in akhir_cols if c in df_results.columns]
            df_results = df_results[cols]
        return df_results, saldo_awal, saldo_awal_b
    
    def unmatched_links(self, df, max_group = 4):
        df = df.copy()
        df["ID"] = '-' 
        unmatched = df[df["Status"] == "Unmatched"]
        group_counter = 1
        used_idx  = set()
        for i, row_i in unmatched.iterrows():
            if i in used_idx:
                continue
            
            target_credit = row_i["Kredit (BB)"]
            target_debit  = row_i["Debit (BB)"]
            date_i        = row_i["Tanggal (BB)"] if target_credit > 0 or target_debit > 0 else row_i["Tanggal (RK)"]
            if target_credit > 0:
                candidates = []
                for r in range(1, max_group+1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx], r
                    ):
                        total = df.loc[list(combo), "Debit (RK)"].sum()
                        if abs(target_credit - total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (RK)"]
                                if abs((date_i - date_j).days) > self.date_tolerance:
                                    valid_dates = False
                                    break
                            if valid_dates:
                                candidates = list(combo)
                                break
                    if candidates:
                        break
                if candidates:
                    group_id = f"G{group_counter}"
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter += 1
                    continue    
            
            # CASE 2 Kredit Rekening > 0, cari kombinasi Debit Jurnal
            elif row_i["Kredit (RK)"] > 0:
                candidates = []            
                for r in range(1, max_group + 1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx],
                        r
                    ):
                        total = df.loc[list(combo), "Debit (BB)"].sum()
                        if abs(row_i["Kredit (RK)"] - total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (BB)"]
                                if abs((date_i - date_j).days) > self.date_tolerance:
                                    valid_dates = False
                                    break
                            if valid_dates:
                                candidates = list(combo)
                                break
                    if candidates:
                        break
                if candidates:
                    group_id = f"G{group_counter}"
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter += 1
            
            # CASE 3: Debit Rekening > 0, cari kombinasi Kredit Jurnal
            elif row_i["Debit (RK)"] > 0:
                candidates = []
                for r in range(1, max_group + 1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx],
                        r
                    ):
                        total = df.loc[list(combo), "Kredit (BB)"].sum()
                        if abs(row_i["Debit (RK)"]-total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (BB)"]
                                if abs((date_i - date_j).days) > self.date_tolerance:
                                    valid_dates = False
                                    break
                            if valid_dates:
                                candidates = list(combo)
                                break
                    if candidates:
                        break
                if candidates:
                    group_id = f"G{group_counter}"
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter += 1
                    
            # CASE 4: Debit Jurnal > 0, cari kombinasi Kredit Rekening
            elif row_i["Debit (BB)"] > 0:
                candidates = []
                for r in range(1, max_group + 1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx],
                        r
                    ):
                        total = df.loc[list(combo), "Kredit (RK)"].sum()
                        if abs(row_i["Debit (BB)"] - total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (RK)"]
                                if abs((date_i - date_j).days) > self.date_tolerance:
                                    valid_dates = False
                                    break
                            if valid_dates:
                                candidates = list(combo)
                                break
                    if candidates:
                        break
                if candidates:
                    group_id = f"G{group_counter}"
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter += 1 

        df["Tanggal Referensi"] = df["Tanggal (BB)"].fillna(df["Tanggal (RK)"])
        df["HasID"] = (df["ID"] != "-")
        df["ID_num"] = pd.to_numeric(df["ID"].str.extract(r"G(\d+)")[0], errors="coerce").fillna(np.inf)

        df = df.sort_values(
            by=["Tanggal Referensi", "HasID", "ID_num"],
            ascending=[True, False, True],
            kind="mergesort" 
        ).drop(columns=["Tanggal Referensi", "HasID", "ID_num"])   
        return df, group_counter
    
    def add_unmatched_links(self, df, max_group=1):
        df = df.copy()
        unmatched = df[(df["Status"] == "Unmatched") & (df["ID"] == "-")]
        
        used_idx = set()
        group_counter_bb = 1
        group_counter_rk = 1
        
        def _find_candidates(idx, row, col_debit, col_kredit, col_tanggal):
            date_i = row[col_tanggal] if pd.notna(row[col_tanggal]) else None
            for r in range(1, max_group + 1):
                for combo in itertools.combinations(
                    [j for j in unmatched.index if j != idx and j not in used_idx], r
                ):
                    total = df.loc[list(combo), col_kredit].sum()
                    if abs(row[col_debit] - total) <= self.amount_tolerance:
                        valid_dates = True
                        if date_i is not None:
                            for j in combo:
                                date_j = df.loc[j, col_tanggal]
                                if pd.notna(date_j) and abs((date_i - date_j).days) > self.date_tolerance:
                                    valid_dates = False
                                    break
                            if valid_dates:
                                return list(combo)
            return []

        for i, row_i in unmatched.iterrows():
            if i in used_idx:
                continue
            
            # ---- Case 1: Debit (BB) & Kredit (BB) ----
            if row_i["Debit (BB)"] > 0:
                candidates = _find_candidates(i, row_i, "Debit (BB)", "Kredit (BB)", "Tanggal (BB)")
                if candidates:
                    group_id = f'BB{group_counter_bb}'
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter_bb += 1
                    continue
                
            elif row_i["Kredit (BB)"] > 0:
                candidates = _find_candidates(i, row_i, "Kredit (BB)", "Debit (BB)", "Tanggal (BB)")
                if candidates:
                    group_id = f'BB{group_counter_bb}'
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter_bb += 1
                    continue
                
            # ---- Case 2: Debit (Rekening) & Kredit (Rekening) ----
            if row_i["Debit (RK)"] > 0:
                candidates = _find_candidates(i, row_i, "Debit (RK)", "Kredit (RK)", "Tanggal (RK)")
                if candidates:
                    group_id = f'RK{group_counter_rk}'
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter_rk += 1
                    continue
                
            elif row_i["Kredit (RK)"] > 0:
                candidates = _find_candidates(i, row_i, "Kredit (RK)", "Debit (RK)", "Tanggal (RK)")
                if candidates:
                    group_id = f'RK{group_counter_rk}'
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter_rk += 1
                    continue
        return df
    
    def clean_empty_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(df[(df["Tanggal (BB)"].isna()) & (df["Tanggal (RK)"].isna())].index)
        df = df.reset_index(drop=True)
        return df                   

    def apply_saldo(self, df: pd.DataFrame, saldo_awal_bb: float, saldo_awal_b: float, rounding: int = 2) -> pd.DataFrame:
        df = df.copy()
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
                debit = df.iloc[i]["Debit (BB)"]
                kredit = df.iloc[i]["Kredit (BB)"]
                saldo += debit - kredit
                saldo_list[i] = saldo
            df["Saldo (BB)"] = saldo_list

        if "Saldo (RK)" in df.columns and {"Debit (RK)", "Kredit (RK)"}.issubset(df.columns):
            saldo = saldo_awal_b
            saldo_list = df["Saldo (RK)"].tolist()
            for i in range(len(df)):
                debit = df.iloc[i]["Debit (RK)"]
                kredit = df.iloc[i]["Kredit (RK)"]
                saldo += kredit - debit
                saldo_list[i] = saldo
            df["Saldo (RK)"] = saldo_list

        opening_row = {
            "Tanggal (BB)": opening_date_bb, 
            "No Voucher": "-", 
            "Debit (BB)": 0.0,
            "Kredit (BB)": 0.0, "Saldo (BB)": saldo_awal_bb,
            "Tanggal (RK)": opening_date_rk, "Debit (RK)": 0.0,
            "Kredit (RK)": 0.0, "Saldo (RK)": saldo_awal_b,
            "Debit (BB) - Kredit (RK)" : "-",
            "Kredit (BB) - Debit (RK)" : "-",
            "Status": "Opening Balance",
            "ID": "-",
            "Catatan": "-"
        }
        df = pd.concat([pd.DataFrame([opening_row]), df], ignore_index=True)

        total_debit_bb = df["Debit (BB)"].sum()
        total_kredit_bb = df["Kredit (BB)"].sum()
        saldo_bb_akhir = round(saldo_awal_bb + total_debit_bb - total_kredit_bb, rounding)

        total_debit_b = df["Debit (RK)"].sum()
        total_kredit_b = df["Kredit (RK)"].sum()
        saldo_b_akhir = round(saldo_awal_b + total_kredit_b - total_debit_b, rounding)

        closing_row = {
            "Tanggal (BB)": closing_date_bb, 
            "No Voucher": "-", 
            "Debit (BB)": total_debit_bb,
            "Kredit (BB)": total_kredit_bb, "Saldo (BB)": saldo_bb_akhir,
            "Tanggal (RK)": closing_date_rk, "Debit (RK)": total_debit_b,
            "Kredit (RK)": total_kredit_b, "Saldo (RK)": saldo_b_akhir,
            "Debit (BB) - Kredit (RK)" : round(total_debit_bb - total_kredit_b, rounding),
            "Kredit (BB) - Debit (RK)" : round(total_kredit_bb - total_debit_b, rounding),
            "Status": "Closing Balance",
            "ID": "-",
            "Catatan": "-"
        }
        df = pd.concat([df, pd.DataFrame([closing_row])], ignore_index=True)
        df_summary = pd.DataFrame([
            {
                "Jenis Saldo": "Saldo Awal",
                "Tanggal": opening_date_bb,
                "Buku Besar (BB)": saldo_awal_bb,
                "Rekening Koran (RK)": saldo_awal_b,
                "Selisih": round(saldo_awal_bb - saldo_awal_b, rounding),
            },
            {
                "Jenis Saldo": "Saldo Akhir",
                "Tanggal": closing_date_bb,
                "Buku Besar (BB)": saldo_bb_akhir,
                "Rekening Koran (RK)": saldo_b_akhir,
                "Selisih": round(saldo_bb_akhir - saldo_b_akhir, rounding),
            },
        ])
        return df, df_summary

    def format_nominal(self, x, column_name=None) -> str:
        try:
            if pd.isna(x):
                return "-"
            x = float(x)
            if x == 0 and column_name not in ["Saldo (RK)", "Saldo (BB)", "Debit (RK)", 
                                              "Kredit (RK)", "Debit (BB)", "Kredit (BB)", "Debit (BB) - Kredit (RK)",
                                              "Kredit (BB) - Debit (RK)", "Selisih"]:
                return "-"
            s = f"{x:,.2f}"
            s = s.replace(",", "X").replace(".", ",").replace("X", ".")
            return s
        except (ValueError, TypeError):
            return "-"

    def is_blank(self, x)->bool:
        if pd.isna(x): return True
        x = str(x).strip().lower()
        return x in {"", "-", "nan", "na", "null", "none"}
    
    def near_enough(self, a, b, abs_tol=1.0, rel_tol=0.0001):
        m = max(abs(a), abs(b))
        tol = max(abs_tol, rel_tol * m)
        return abs(a-b) <= tol

    def get_next_counter (self, df, col="ID"):
        max_num = 0
        if col in df.columns:
            for val in df[col].dropna().astype(str):
                m = re.match(r"^G\s*(\d+)$", val.strip())
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
        return max_num + 1
        
    def final_trace(self, df, max_group=4):
        df = df.copy()
        counter = self.get_next_counter(df, col="ID")
        
        if "Catatan" not in df.columns:
            df["Catatan"] = "-"

        closing_db_cr = df["Debit (BB) - Kredit (RK)"].iloc[-1]
        closing_cr_db = df["Kredit (BB) - Debit (RK)"].iloc[-1]

        if closing_db_cr == closing_cr_db:
            mask = (df["Status"] == "Unmatched") & ((df["ID"] == "-") | (df["ID"].isna()))
            df.loc[mask, "ID"] = f"G{counter}"
            return df

        selisih = abs(closing_db_cr - closing_cr_db)
        candidates = df[(df["Status"] == "Unmatched") & ((df["ID"] == "-") | (df["ID"].isna()))]
        candidate_idx = candidates.index.tolist()

        found = False
        for r in range(1, max_group+1):
            for combo in itertools.combinations(candidate_idx, r):
                total = abs(df.loc[list(combo), "Debit (BB) - Kredit (RK)"].sum() -
                        df.loc[list(combo), "Kredit (BB) - Debit (RK)"].sum())
                if abs(total - selisih) < 1e-6:
                    bb_only, rk_only = [], []
                    for idx in combo:
                        bb_amt = max(df.at[idx, "Debit (BB)"], df.at[idx, "Kredit (BB)"])
                        rk_amt = max(df.at[idx, "Debit (RK)"], df.at[idx, "Kredit (RK)"])
                        if bb_amt > 0 and rk_amt == 0: bb_only.append(idx)
                        if rk_amt > 0 and bb_amt == 0: rk_only.append(idx)
                        
                    paired = set()
                    for i in rk_only:
                        amt_rk = df.at[i, "Debit (RK)"] or df.at[i, "Kredit (RK)"]
                        best   = None
                        for j in bb_only:
                            if j in paired:
                                continue
                            amt_bb = df.at[j, "Debit (BB)"] or df.at[j, "Kredit (BB)"]
                            if self.near_enough(amt_rk, amt_bb):
                                best = j
                                df.at[j, "ID"] = df.at[i, "ID"] = f"G{counter}"
                        if best is not None:
                            vch_i = df.at[i, "No Voucher"] if "No Voucher" in df.columns else None
                            vch_j = df.at[best, "No Voucher"] if "No Voucher" in df.columns else None
                            voucher = vch_i if not self.is_blank(vch_i) else (vch_j if not self.is_blank(vch_j) else None)

                            df.at[i, "Catatan"] = (str(voucher) if voucher else "Typo (cek nilai & bukti)")
                            df.at[best, "Catatan"] = (str(voucher) if voucher else "Typo (cek nilai & bukti)")

                            paired.add(i); paired.add(best)
                            counter += 1
                            break

                    for idx in combo:
                        if idx in paired:
                            continue
                        v = df.at[idx, "No Voucher"] if "No Voucher" in df.columns else None
                        if not self.is_blank(v):
                            df.at[idx, "Catatan"] = str(v)
                        else:
                            df.at[idx, "Catatan"] = "Tambahkan Jurnal"

                    found = True
                    break
            if found:
                break

        if not found and candidate_idx:
            for idx in candidate_idx:
                voucher = df.at[idx, "No Voucher"]
                if voucher in [None, "-", "nan", "NaN", ""]:
                    df.at[idx, "Catatan"] = "Rekening Koran"
                else:
                    df.at[idx, "Catatan"] = f"{voucher}"
        return df

    def matching(self):
        self.load_data()
        self.journal_df = self.preprocess(self.journal_df)
        self.bank_df    = self.preprocess(self.bank_df)
        self.matched_df, saldo_awal_bb, saldo_awal_b = self.greedy_matching(self.journal_df, self.bank_df)
        self.matched_df, counter = self.unmatched_links(self.matched_df)
        
        self.matched_df = self.add_unmatched_links(self.matched_df)
        self.matched_df = self.clean_empty_dates(self.matched_df)
        self.matched_df, self.df_summary = self.apply_saldo(self.matched_df, saldo_awal_bb, saldo_awal_b)
        self.matched_df = self.final_trace(self.matched_df)

        with pd.ExcelWriter(self.output_path, engine="openpyxl") as writer:
            self.matched_df.to_excel(writer, sheet_name="Transaksi", index=False)
            self.df_summary.to_excel(writer, sheet_name="Summary", index=False)
        print(f"Hasil matching & summary disimpan di: {self.output_path}")
        return self.matched_df