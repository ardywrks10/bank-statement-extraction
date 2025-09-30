import pandas as pd
import numpy as np
import re
from datetime import date, datetime

def _to_iso_date(x):
    try:
        if x is None or pd.isna(x):
            return None
    except Exception:
        if x is None:
            return None
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().strftime("%Y-%m-%d")
    if isinstance(x, np.datetime64):
        dt = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime().strftime("%Y-%m-%d")
    if isinstance(x, (datetime, date)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, str):
        s = x.strip()
        return None if s in {"", "-"} else s
    return None

def _snake(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"__+", "_", s)
    return s

TRANSACTION_KEY_ORDER = [
    "tanggal_bb","no_voucher","debit_bb","kredit_bb","saldo_bb",
    "tanggal_rk","debit_rk","kredit_rk","saldo_rk",
    "debit_bb_kredit_rk","kredit_bb_debit_rk",
    "status","id_jurnal","catatan"
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
    "Jurnal ID": "id_jurnal",
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
    rename_map = {c: TRANSACTION_COL_MAP.get(c, _snake(str(c))) for c in df.columns}
    dfr = df.rename(columns=rename_map).copy()

    # (opsional) kalau ada kolom ganda 'id_jurnal', koales jadi satu nilai
    if (dfr.columns == "id_jurnal").sum() > 1:
        sub = dfr.loc[:, dfr.columns == "id_jurnal"]
        def first_valid(row):
            for x in row:
                if x is None: continue
                if isinstance(x, float) and pd.isna(x): continue
                if isinstance(x, str) and x.strip() in {"", "-"}: continue
                return x
            return None
        dfr["id_jurnal"] = sub.apply(first_valid, axis=1)
        # buang kolom duplikat, sisakan satu
        dfr = pd.concat([dfr.loc[:, dfr.columns != "id_jurnal"], dfr[["id_jurnal"]]], axis=1)

    recs = []
    for _, row in dfr.iterrows():
        obj = {}
        for k in TRANSACTION_KEY_ORDER:
            if k in dfr.columns:
                v = row.get(k)

                # jika v keisengan berupa Series, ambil first valid
                if isinstance(v, pd.Series):
                    vv = None
                    for x in v.tolist():
                        if x is None: continue
                        if isinstance(x, float) and pd.isna(x): continue
                        if isinstance(x, str) and x.strip() in {"", "-"}: continue
                        vv = x; break
                    v = vv

                if k in ("tanggal_bb","tanggal_rk"):
                    obj[k] = _to_iso_date(v)
                elif k == "id_jurnal":
                    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() in {"", "-"}):
                        obj[k] = None
                    else:
                        obj[k] = str(v)
                else:
                    if isinstance(v, str) and v.strip() in {"", "-"}:
                        obj[k] = None
                    else:
                        obj[k] = None if pd.isna(v) else v
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
                    obj[k] = None if pd.isna(v) else v
        recs.append(obj)
    return recs
