from fastapi import APIRouter, Depends, Form, HTTPException
from typing import Optional, List
from app.dto.bank_dto import BankConfigIn, BankConfigUpdate, BankListOut, BankCreateOut, BankUpdateOut, BankDeleteOut
from app.services.bank_service import BankService

router = APIRouter(prefix="/bank", tags=["bank"])

def parse_headers_csv(headers_csv: str) -> List[str]:
    items = [h.strip() for h in (headers_csv or "").split(",") if h.strip()]
    if not items:
        raise HTTPException(status_code=422, detail="HEADERS wajib diisi (CSV)")
    return items

@router.get("/list", response_model=BankListOut)
def list_banks(service: BankService = Depends(BankService)):
    return service.list()

@router.post("/create", response_model=BankCreateOut)
def create_bank(
    name: str = Form(...),
    HEADERS: str = Form(...),
    keterangan: Optional[str] = Form(None),
    kolom_kode: Optional[str] = Form(None),
    target_kode: Optional[str] = Form(None),
    debit_code: Optional[str] = Form(None),
    kredit_code: Optional[str] = Form(None),
    header_per_page: Optional[bool] = Form(True),
    DATE_FORMAT: str = Form(...),
    service: BankService = Depends(BankService),
):
    cfg = BankConfigIn(
        name=name,
        HEADERS=parse_headers_csv(HEADERS),
        keterangan=keterangan,
        kolom_kode=kolom_kode,
        target_kode=target_kode,
        debit_code=debit_code,
        kredit_code=kredit_code,
        header_per_page=bool(header_per_page),
        DATE_FORMAT=DATE_FORMAT,
    )
    return service.create(cfg)

@router.delete("/delete/{name}", response_model=BankDeleteOut)
def delete_bank(name: str, service: BankService = Depends(BankService)):
    return service.delete(name)

@router.patch("/update/{bank_name}", response_model=BankUpdateOut)
def update_bank(
    bank_name: str,
    name: Optional[str] = Form(None),
    HEADERS: Optional[str] = Form(None),
    keterangan: Optional[str] = Form(None),
    kolom_kode: Optional[str] = Form(None),
    target_kode: Optional[str] = Form(None),
    debit_code: Optional[str] = Form(None),
    kredit_code: Optional[str] = Form(None),
    header_per_page: Optional[bool] = Form(None),
    DATE_FORMAT: Optional[str] = Form(None),
    service: BankService = Depends(BankService),
):
    patch = BankConfigUpdate(
        name=name,
        HEADERS=parse_headers_csv(HEADERS) if HEADERS is not None else None,
        keterangan=keterangan,
        kolom_kode=kolom_kode,
        target_kode=target_kode,
        debit_code=debit_code,
        kredit_code=kredit_code,
        header_per_page=header_per_page if header_per_page is not None else None,
        DATE_FORMAT=DATE_FORMAT,
    )
    return service.update(bank_name, patch)
