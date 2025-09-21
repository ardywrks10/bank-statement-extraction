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
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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