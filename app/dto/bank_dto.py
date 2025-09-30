from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

class BankConfigIn(BaseModel):
    name: str = Field(..., example="BCA")
    HEADERS: List[str] = Field(..., description="Header kolom transaksi")
    keterangan: Optional[str] = None
    kolom_kode: Optional[str] = None
    target_kode: Optional[str] = None
    debit_code: Optional[str] = None
    kredit_code: Optional[str] = None
    header_per_page: bool = True
    DATE_FORMAT: str = Field(..., example="dd-MM-yyyy")

    @field_validator("name")
    @classmethod
    def safe_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Field 'name' wajib diisi")
        if not _NAME_RE.match(v):
            raise ValueError("Gunakan huruf/angka/underscore/dash saja")
        return v

class BankConfigUpdate(BaseModel):
    name: Optional[str] = None
    HEADERS: Optional[List[str]] = None
    keterangan: Optional[str] = None
    kolom_kode: Optional[str] = None
    target_kode: Optional[str] = None
    debit_code: Optional[str] = None
    kredit_code: Optional[str] = None
    header_per_page: Optional[bool] = None
    DATE_FORMAT: Optional[str] = None

    @field_validator("name")
    @classmethod
    def safe_name_update(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip()
        if not v2:
            raise ValueError("Field 'name' tidak boleh kosong")
        if not _NAME_RE.match(v2):
            raise ValueError("Gunakan huruf/angka/underscore/dash saja")
        return v2

class BankListOut(BaseModel):
    banks: List[str]

class BankCreateOut(BaseModel):
    message: str
    bank_key: str

class BankUpdateOut(BaseModel):
    message: str
    bank_key: str
    changed_fields: List[str]
    before: dict
    after: dict

class BankDeleteOut(BaseModel):
    message: str
    unregistered: bool
