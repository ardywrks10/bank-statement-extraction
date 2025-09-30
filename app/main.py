from fastapi import FastAPI
from contextlib import asynccontextmanager
import easyocr

from app.routes import process_route, bank_route
from app.extractors import dynamic_registry as reg

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init OCR sekali di startup
    reader = easyocr.Reader(["id", "en"])
    reg.set_reader(reader)
    reg.load_all_on_startup()
    yield
    # tempat cleanup kalau perlu

app = FastAPI(
    title="Bank Statement Extraction & Matching API",
    lifespan=lifespan,
)

# Daftarkan routers
app.include_router(process_route.router)
app.include_router(bank_route.router)
