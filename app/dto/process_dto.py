from pydantic import BaseModel
from typing import Optional, List

class TransactionOut(BaseModel):
    tanggal_bb: Optional[str] = None
    no_voucher: Optional[str] = None
    debit_bb: Optional[float] = None
    kredit_bb: Optional[float] = None
    saldo_bb: Optional[float] = None
    tanggal_rk: Optional[str] = None
    debit_rk: Optional[float] = None
    kredit_rk: Optional[float] = None
    saldo_rk: Optional[float] = None
    status: Optional[str] = None
    id_jurnal: Optional[str] = None
    catatan: Optional[str] = None

class SummaryOut(BaseModel):
    jenis_saldo: Optional[str] = None
    tanggal: Optional[str] = None
    buku_besar_bb: Optional[float] = None
    rekening_koran_rk: Optional[float] = None
    selisih: Optional[float] = None

class ProcessOut(BaseModel):
    status: str
    bank: str
    periode_id: str
    id_perkiraan: str
    id_department: str
    transactions: List[TransactionOut]
    summary: List[SummaryOut]
    rows_extracted: int
    rows_matched: int
    output_excel_bank: str
    output_excel_bb: str
    output_matching: str
