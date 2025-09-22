import re
import itertools
import pandas as pd
from typing import Iterable, List, Optional

class Reconciler:
    def __init__(self, abs_tol: float = 1.0, rel_tol: float = 1e-4, max_group: int = 4):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_group = max_group
        
    @staticmethod
    def _is_blank(x) -> bool:
        if pd.isna(x):
            return True
        x = str(x).strip().lower()
        return x in {"", "-", "nan", "na", "null", "none"}
    
    def _near_enough(self, a: float, b: float) -> bool:
        a = float(a)
        b = float(b)
        m = max(abs(a), abs(b))
        tol = max(self.abs_tol, self.rel_tol * m)
        return abs(a - b) <= tol
    
    @staticmethod
    def _max_two(a, b) -> float:
        a = 0.0 if pd.isna(a) else float(a)
        b = 0.0 if pd.isna(b) else float(b)
        return max(a, b)
    
    @staticmethod
    def _get_next_counter(df: pd.DataFrame, col: str = "ID") -> int:
        max_num = 0
        if col in df.columns:
            for val in df[col].dropna().astype(str):
                m = re.match(r"^G\s*(\d+)$", val.strip())
                if m:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
        return max_num + 1
    
    def rematching(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df.copy() if isinstance(df, pd.DataFrame) else df
        
        df = df.copy()
        # Validasi kolom tabel
        for c, v in {"ID": "-", "Status": "", "Debit (BB)": 0.0, "Kredit (BB)": 0.0,
                     "Debit (RK)": 0.0, "Kredit (RK)": 0.0, 
                     "Catatan": "-"}.items():
            if c not in df.columns:
                df[c] = v
        
        for c in ["Debit (BB)", "Kredit (BB)", "Debit (RK)", "Kredit (RK)"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        
        status_col   = df["Status"]
        blank_status = status_col.isna() | status_col.astype(str).str.strip().str.lower().isin(["", "-", "unmatched"])
        blank_id     = df["ID"].isna() | df["ID"].astype(str).str.strip().isin(["", "-"])
        cand_mask    = blank_status & blank_id 
        cand_idx     = df.index[cand_mask].tolist()
        
        rk_debit     = [i for i in cand_idx if self._max_two(df.at[i, "Debit (RK)"], 0) > 0 and self._max_two(df.at[i, "Kredit (RK)"], 0) == 0]
        rk_kredit    = [i for i in cand_idx if self._max_two(df.at[i, "Kredit (RK)"], 0) > 0 and self._max_two(df.at[i, "Debit (RK)"], 0) == 0]   
        bb_debit     = [i for i in cand_idx if self._max_two(df.at[i, "Debit (BB)"], 0) > 0 and self._max_two(df.at[i, "Kredit (BB)"], 0) == 0]
        bb_kredit    = [i for i in cand_idx if self._max_two(df.at[i, "Kredit (BB)"], 0) > 0 and self._max_two(df.at[i, "Debit (BB)"], 0) == 0]
        
        bb_debit.sort()
        bb_kredit.sort()
        rk_debit.sort()
        rk_kredit.sort()
        
        used      = set()
        counter   = self._get_next_counter(df, col="ID")
        def _amount(idx, side_main, side_alt):
            return self._max_two(df.at[idx, side_main], df.at[idx, side_alt])
        
        def _try_match(anchor_idx:int, pool: list, pool_side_main: str, pool_side_alt: str)-> bool:
            nonlocal counter
            if anchor_idx in used:
                return False
            
            if df.at[anchor_idx, "Debit (BB)"] > 0 and df.at[anchor_idx, "Kredit (BB)"] == 0:
                anchor_amt = float(df.at[anchor_idx, "Debit (BB)"])
            else:
                anchor_amt = float(df.at[anchor_idx, "Kredit (BB)"])
            if anchor_amt <= 0:
                return False
            
            avail = [j for j in pool if j not in used and j != anchor_idx]
            avail = [j for j in avail if _amount(j, pool_side_main, pool_side_alt) <= anchor_amt]

            for r in range(1, min(self.max_group, len(avail)) + 1):
                for combo in itertools.combinations(avail, r):
                    s = sum(_amount(j, pool_side_main, pool_side_alt) for j in combo)
                    if s == anchor_amt:
                        gid = f"G{counter}"
                        df.at[anchor_idx, "ID"] = gid
                        df.at[anchor_idx, "Status"] = "Matched"
                        df.at[anchor_idx, "Catatan"] = "-"
                        used.add(anchor_idx)
                        
                        for j in combo:
                            df.at[j, "ID"] = gid
                            df.at[j, "Status"] = "Matched"
                            df.at[j, "Catatan"]= "-"
                            used.add(j)
                        counter += 1
                        return True
            return False
        
        for i in bb_debit: _try_match(i, rk_kredit, "Kredit (RK)", "Debit (RK)")
        for i in bb_kredit: _try_match(i, rk_debit, "Debit (RK)", "Kredit (RK)")
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = self.rematching(df)
        counter = self._get_next_counter(df, col="ID")
        
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
        for r in range(1, self.max_group+1):
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
                        amt_rk = self._max_two(df.at[i, "Debit (RK)"], df.at[i, "Kredit (RK)"])
                        best = None
                        for j in bb_only:
                            if j in paired:
                                continue
                            amt_bb = self._max_two(df.at[j, "Debit (BB)"], df.at[j, "Kredit (BB)"])
                            if self._near_enough(amt_rk, amt_bb):
                                best = j
                                break  
                        if best is not None:
                            df.at[best, "ID"] = df.at[i, "ID"] = f"G{counter}"
                            vch_i = df.at[i, "No Voucher"] if "No Voucher" in df.columns else None
                            vch_j = df.at[best, "No Voucher"] if "No Voucher" in df.columns else None
                            voucher = vch_i if not self._is_blank(vch_i) else (vch_j if not self._is_blank(vch_j) else None)
                            note = str(voucher) if voucher else "Typo (cek nilai & bukti)"
                            df.at[i, "Catatan"] = note
                            df.at[best, "Catatan"] = note
                            paired.add(i); paired.add(best)
                            counter += 1

                    for idx in combo:
                        if idx in paired:
                            continue
                        v = df.at[idx, "No Voucher"] if "No Voucher" in df.columns else None
                        if not self._is_blank(v):
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