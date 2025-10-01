from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.services.process_service import ProcessService
from fastapi.responses import FileResponse
from app.dto.process_dto import ProcessOut
from app.dto.process_dto import ReMatchOut
from app.paths import MATCHING_DIR

router = APIRouter(prefix="/process", tags=["process"])

@router.post("", response_model=ProcessOut)
async def process_file(
    bank_name: str = Form(...),
    pdf_file: UploadFile = File(...),
    pages_mode: str = Form(...),
    pages: str | None = Form(None),
    periode_id: str = Form(...),
    id_perkiraan: str = Form(...),
    id_department: str = Form(...),
    service: ProcessService = Depends(ProcessService),
):
    return service.run(
        bank_name=bank_name,
        pdf_file=pdf_file,
        pages_mode=pages_mode,
        pages=pages,
        periode_id=periode_id,
        id_perkiraan=id_perkiraan,
        id_department=id_department,
    )

@router.post("/rematch", response_model=ReMatchOut)
def process_match_again(
    bank_name: str = Form(...),
    periode_id: str = Form(...),
    id_perkiraan: str = Form(...),
    id_department: str = Form(...),
    service: ProcessService = Depends(ProcessService),
):
    return service.rematch(
        bank_name=bank_name,
        periode_id=periode_id,
        id_perkiraan=id_perkiraan,
        id_department=id_department,
        save_excel=True,
    )


@router.get("/download/{bank_name}")
def download_matching(bank_name: str):
    """
    Unduh file bukti matching hasil terakhir.
    Contoh: GET /process/download/bni  -> data/bukti_matching/bni_matching.xlsx
    """
    path = MATCHING_DIR / f"{bank_name}_matching.xlsx"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )
