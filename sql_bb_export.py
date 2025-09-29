# sql_bb_export.py
import os
import pymysql
import pandas as pd
from datetime import date
from typing import Optional, List, Dict, Any
import decimal

# --- DB config (hardcode sesuai permintaanmu) ---
MYSQL_CFG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "sedana_raditya",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
}

SQL_SALDO = """
SELECT 
    periode_id, id_perkiraan, id_department,
    debet, kredit, kurs, mata_uang
FROM sedana_raditya.aiso_saldo_akhir_perd
WHERE periode_id   = %s
  AND id_perkiraan = %s
  AND id_department= %s
LIMIT 1;
"""

SQL_TRX = """
SELECT 
    JU.JURNAL_ID           AS jurnal_id,
    JU.TGL_TRANSAKSI       AS tgl,
    JU.NO_VOUCHER          AS no_voucher,
    JU.PERIODE_ID          AS periode_id,
    JU.DEPARTMENT_ID       AS departement_id,
    JU.KETERANGAN          AS ket,
    SUM(JD.DEBET)          AS debit,
    SUM(JD.KREDIT)         AS kredit
FROM aiso_jurnal_umum JU
JOIN aiso_jurnal_detail JD 
  ON JU.JURNAL_ID = JD.JURNAL_ID
WHERE JD.ID_PERKIRAAN = %s
  AND JU.PERIODE_ID   = %s
  AND JU.DEPARTMENT_ID= %s
GROUP BY 
  JU.TGL_TRANSAKSI, JU.NO_VOUCHER, JU.KETERANGAN, JD.NOTE, JU.JURNAL_ID
ORDER BY 
  JU.TGL_TRANSAKSI, JU.NO_VOUCHER;
"""

def _prev_periode_id(periode_id: str) -> str:
    s = str(periode_id)
    if len(s) != 9 or not s.startswith("1"):
        raise ValueError("Format periode_id invalid. Contoh valid: 120240601")
    year = int(s[1:5]); month = int(s[5:7])
    if month == 1: year -= 1; month = 12
    else: month -= 1
    return f"1{year:04d}{month:02d}01"

def _jsonable(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out = {}
    for k, v in row.items():
        if isinstance(v, decimal.Decimal):
            out[k] = float(v)
        elif isinstance(v, (date, )):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out

def _fetch_one(sql: str, params: tuple) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = pymysql.connect(**MYSQL_CFG)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        try:
            if conn: conn.close()
        except Exception:
            pass

def _fetch_all(sql: str, params: tuple) -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = pymysql.connect(**MYSQL_CFG)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        try:
            if conn: conn.close()
        except Exception:
            pass

def build_journal_dataframe(opening_row: Optional[Dict[str, Any]],
                            trx_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Header final (persis yang kamu minta):
    ["Nama Akun / Tanggal","Jurnal ID","Nomor","Keterangan","Debit","Kredit","Saldo"]
    """
    cols = ["Nama Akun / Tanggal", "Jurnal ID", "Nomor", "Keterangan", "Debit", "Kredit", "Saldo"]
    rows = []
    opening_debet = float(opening_row.get("debet", 0.0)) if opening_row else 0.0

    # SALDO AWAL
    rows.append({
        "Nama Akun / Tanggal": None,
        "Jurnal ID": "-",
        "Nomor": "SALDO AWAL",
        "Keterangan": "SALDO AWAL",
        "Debit": 0.0,
        "Kredit": 0.0,
        "Saldo": opening_debet,
    })

    total_debit = 0.0
    total_kredit = 0.0

    # Transaksi (split debit/kredit seperti format lama)
    for r in trx_rows:
        tgl = r.get("tgl")
        try:
            tgl = pd.to_datetime(tgl, errors="coerce").date() if tgl else None
        except Exception:
            tgl = None

        jurnal_id = str(r.get("jurnal_id") or "")
        vch       = r.get("no_voucher") or ""
        ket       = r.get("ket") or ""
        deb       = float(r.get("debit") or 0.0)
        kre       = float(r.get("kredit") or 0.0)

        if deb > 0:
            rows.append({
                "Nama Akun / Tanggal": tgl,
                "Jurnal ID": jurnal_id,
                "Nomor": vch,
                "Keterangan": ket,
                "Debit": deb,
                "Kredit": 0.0,
                "Saldo": 0.0,
            })
            total_debit += deb

        if kre > 0:
            rows.append({
                "Nama Akun / Tanggal": tgl,
                "Jurnal ID": jurnal_id,
                "Nomor": vch,
                "Keterangan": ket,
                "Debit": 0.0,
                "Kredit": kre,
                "Saldo": 0.0,
            })
            total_kredit += kre

    # SALDO AKHIR (Total)
    closing_suggest = round(opening_debet + total_debit - total_kredit, 2)
    rows.append({
        "Nama Akun / Tanggal": None,
        "Jurnal ID": "-",
        "Nomor": "SALDO AKHIR",
        "Keterangan": "SALDO AKHIR",
        "Debit": total_debit,
        "Kredit": total_kredit,
        "Saldo": closing_suggest,
    })

    return pd.DataFrame(rows, columns=cols)

def export_bb_excel_from_sql(periode_id: str, id_perkiraan: str, id_department: str, out_path: str) -> str:
    """
    Ambil opening (periode-1), transaksi (periode), susun DF dengan header target dan simpan ke Excel.
    return: path Excel (out_path)
    """
    periode_awal = _prev_periode_id(periode_id)
    opening = _jsonable(_fetch_one(SQL_SALDO, (periode_awal, id_perkiraan, id_department)))
    trx = [_jsonable(r) for r in _fetch_all(SQL_TRX, (id_perkiraan, periode_id, id_department))]

    df = build_journal_dataframe(opening, trx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path
