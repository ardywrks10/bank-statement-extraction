import pandas as pd
import numpy as np
import re
import itertools
from datetime import timedelta

class BankJournalMatcher:
    def __init__(self, journal_path, bank_path, output_path="matched_df.xlsx"):
        self.journal_path  = journal_path
        self.bank_path     = bank_path
        self.output_path   = output_path
        self.journal_df    = None
        self.bank_df       = None
        self.matched_df    = None
        self.balance_df    = None
        self.date_tolerance = 7
        self.amount_tolerance = 0.9

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
                df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0))
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
            j_tgl    = j_row["Tgl"]
            j_debit  = j_row["Debit"]
            j_kredit = j_row["Kredit"]

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
                            "Tanggal (BB)"       : j_tgl,
                            "Debit (BB)"         : j_debit,
                            "Kredit (BB)"        : j_kredit,
                            "Saldo (BB)"         : 0.0,
                            "Tanggal (Rekening)" : b_tgl,
                            "Debit (Rekening)"   : b_debet,
                            "Kredit (Rekening)"  : b_kredit,
                            "Saldo (Rekening)"   : 0.0,
                            "Status"             : "Matched"
                        }
                        matched_bank_idxs.add(b_idx)
                        break

            if found:
                results.append(found)
            else:
                results.append({
                    "Tanggal (BB)"       : j_tgl,
                    "Debit (BB)"         : j_debit,
                    "Kredit (BB)"        : j_kredit,
                    "Saldo (BB)"         : 0.0,
                    "Tanggal (Rekening)" : "-",
                    "Debit (Rekening)"   : 0.0,
                    "Kredit (Rekening)"  : 0.0,
                    "Saldo (Rekening)"   : 0.0,
                    "Status"             : "Unmatched"
                })

        for b_idx, b_row in bank_n1.iterrows():
            if b_idx not in matched_bank_idxs:
                b_debet  = b_row["Debit"]
                b_kredit = b_row["Kredit"]
                results.append({
                    "Tanggal (BB)"      : "-",
                    "Debit (BB)"        : 0.0,
                    "Kredit (BB)"       : 0.0,
                    "Saldo (BB)"        : 0.0,
                    "Tanggal (Rekening)": b_row["Tgl"],
                    "Debit (Rekening)"  : b_debet,
                    "Kredit (Rekening)" : b_kredit,
                    "Saldo (Rekening)"  : 0.0,
                    "Status"            : "Unmatched"
                })
                
        saldo_awal   = round(float(journal_df.iloc[0]["Saldo"]), rounding)
        saldo_awal_b = round(float(bank_df.iloc[0]["Saldo"]), rounding)
        df_results = pd.DataFrame(results)
        cols       = ["Tanggal (BB)", "Tanggal (Rekening)"]
        df_results[cols]  = df_results[cols].apply(
            lambda col: pd.to_datetime(col, format="%d/%m/%Y", errors="coerce").dt.date)
        df_results["Tanggal Referensi"] = df_results["Tanggal (BB)"].fillna(df_results["Tanggal (Rekening)"])
        df_results = df_results.sort_values(by=["Tanggal Referensi", "Tanggal (BB)", 
                                                "Tanggal (Rekening)"], ascending=[True, True, True])
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

            date_i        = row_i["Tanggal (BB)"] if target_credit > 0 or target_debit > 0 else row_i["Tanggal (Rekening)"]
            if target_credit > 0:
                candidates = []
                for r in range(1, max_group+1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx], r
                    ):
                        total = df.loc[list(combo), "Debit (Rekening)"].sum()
                        if abs(target_credit - total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (Rekening)"]
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
            elif row_i["Kredit (Rekening)"] > 0:
                candidates = []            
                for r in range(1, max_group + 1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx],
                        r
                    ):
                        total = df.loc[list(combo), "Debit (BB)"].sum()
                        if abs(row_i["Kredit (Rekening)"] - total) <= self.amount_tolerance:
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
            elif row_i["Debit (Rekening)"] > 0:
                candidates = []
                for r in range(1, max_group + 1):
                    for combo in itertools.combinations(
                        [j for j in unmatched.index if j != i and j not in used_idx],
                        r
                    ):
                        total = df.loc[list(combo), "Kredit (BB)"].sum()
                        if abs(row_i["Debit (Rekening)"]-total) <= self.amount_tolerance:
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
                        total = df.loc[list(combo), "Kredit (Rekening)"].sum()
                        if abs(row_i["Debit (BB)"] - total) <= self.amount_tolerance:
                            valid_dates = True
                            for j in combo:
                                date_j = df.loc[j, "Tanggal (Rekening)"]
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

        df["Tanggal Referensi"] = df["Tanggal (BB)"].fillna(df["Tanggal (Rekening)"])
        df["HasID"] = (df["ID"] != "-")
        df["ID_num"] = pd.to_numeric(df["ID"].str.extract(r"G(\d+)")[0], errors="coerce").fillna(np.inf)

        df = df.sort_values(
            by=["Tanggal Referensi", "HasID", "ID_num"],
            ascending=[True, False, True],
            kind="mergesort" 
        ).drop(columns=["Tanggal Referensi", "HasID", "ID_num"])   
        return df 
    
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
            if row_i["Debit (Rekening)"] > 0:
                candidates = _find_candidates(i, row_i, "Debit (Rekening)", "Kredit (Rekening)", "Tanggal (Rekening)")
                if candidates:
                    group_id = f'RK{group_counter_rk}'
                    df.loc[[i] + candidates, "ID"] = group_id
                    used_idx.add(i)
                    used_idx.update(candidates)
                    group_counter_rk += 1
                    continue
                
            elif row_i["Kredit (Rekening)"] > 0:
                candidates = _find_candidates(i, row_i, "Kredit (Rekening)", "Debit (Rekening)", "Tanggal (Rekening)")
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
        df = df.drop(df[(df["Tanggal (BB)"].isna()) & (df["Tanggal (Rekening)"].isna())].index)
        df = df.reset_index(drop=True)
        return df                   
              

    def apply_saldo(self, df: pd.DataFrame, saldo_awal_bb: float, saldo_awal_b: float, rounding: int = 2) -> pd.DataFrame:
        df = df.copy()
        mid_idx = len(df) // 2  

        tgl_mid_bb      = pd.to_datetime(df.iloc[mid_idx]["Tanggal (BB)"], errors="coerce").date()
        opening_date_bb = (tgl_mid_bb.replace(day=1) - pd.Timedelta(days=1))
        closing_date_bb = (tgl_mid_bb + pd.offsets.MonthEnd(0)).date()

        tgl_mid_rk = pd.to_datetime(df.iloc[mid_idx]["Tanggal (Rekening)"], errors="coerce").date()
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


        if "Saldo (Rekening)" in df.columns and {"Debit (Rekening)", "Kredit (Rekening)"}.issubset(df.columns):
            saldo = saldo_awal_b
            saldo_list = df["Saldo (Rekening)"].tolist()
            for i in range(len(df)):
                debit = df.iloc[i]["Debit (Rekening)"]
                kredit = df.iloc[i]["Kredit (Rekening)"]
                saldo += kredit - debit
                saldo_list[i] = saldo
            df["Saldo (Rekening)"] = saldo_list

        opening_row = {
            "Tanggal (BB)": opening_date_bb, "Debit (BB)": 0.0,
            "Kredit (BB)": 0.0, "Saldo (BB)": saldo_awal_bb,
            "Tanggal (Rekening)": opening_date_rk, "Debit (Rekening)": 0.0,
            "Kredit (Rekening)": 0.0, "Saldo (Rekening)": saldo_awal_b,
            "Status": "Opening Balance",
        }
        df = pd.concat([pd.DataFrame([opening_row]), df], ignore_index=True)

        total_debit_bb = df["Debit (BB)"].sum()
        total_kredit_bb = df["Kredit (BB)"].sum()
        saldo_bb_akhir = round(saldo_awal_bb + total_debit_bb - total_kredit_bb, rounding)

        total_debit_b = df["Debit (Rekening)"].sum()
        total_kredit_b = df["Kredit (Rekening)"].sum()
        saldo_b_akhir = round(saldo_awal_b + total_kredit_b - total_debit_b, rounding)

        closing_row = {
            "Tanggal (BB)": closing_date_bb, "Debit (BB)": total_debit_bb,
            "Kredit (BB)": total_kredit_bb, "Saldo (BB)": saldo_bb_akhir,
            "Tanggal (Rekening)": closing_date_rk, "Debit (Rekening)": total_debit_b,
            "Kredit (Rekening)": total_kredit_b, "Saldo (Rekening)": saldo_b_akhir,
            "Status": "Closing Balance",
        }
        df = pd.concat([df, pd.DataFrame([closing_row])], ignore_index=True)
        return df


    def format_nominal(self, x, column_name=None) -> str:
        try:
            if pd.isna(x):
                return "-"
            x = float(x)
            if x == 0 and column_name not in ["Saldo (Rekening)", "Saldo (BB)", "Debit (Rekening)", 
                                              "Kredit (Rekening)", "Debit (BB)", "Kredit (BB)"]:
                return "-"
            s = f"{x:,.2f}"
            s = s.replace(",", "X").replace(".", ",").replace("X", ".")
            return s
        except (ValueError, TypeError):
            return "-"

    def matching(self):
        self.load_data()
        self.journal_df = self.preprocess(self.journal_df)
        self.bank_df    = self.preprocess(self.bank_df)
        self.matched_df, saldo_awal_bb, saldo_awal_b = self.greedy_matching(self.journal_df, self.bank_df)
        self.matched_df = self.unmatched_links(self.matched_df)
        
        self.matched_df = self.add_unmatched_links(self.matched_df)
        self.matched_df = self.clean_empty_dates(self.matched_df)
        self.matched_df = self.apply_saldo(self.matched_df, saldo_awal_bb, saldo_awal_b)
        for col in ["Debit (BB)", "Kredit (Rekening)", "Kredit (BB)", 
                    "Debit (Rekening)", "Saldo (BB)", "Saldo (Rekening)"]:
            if col in self.matched_df.columns:
                self.matched_df[col] = self.matched_df[col].apply(lambda x: self.format_nominal(x, col))
        self.matched_df.to_excel(self.output_path, index=False)
        print(f"Hasil matching disimpan di: {self.output_path}")
        return self.matched_df

