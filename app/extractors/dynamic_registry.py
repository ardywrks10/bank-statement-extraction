from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Optional, Type
from fastapi import HTTPException

from app.extractors.pipeline import Pipeline as DefaultBaseExtractor
from app.extractors.bca import BCAExtractor
from app.extractors.bni import BNIExtractor
from app.extractors.permata import PermataExtractor
from app.extractors.mandiri import MandiriExtractor

BANKS_DIR = Path("banks")
BANKS_DIR.mkdir(parents=True, exist_ok=True)

# ============= State =============
_EASYOCR_READER = None
_DYNAMIC_EXTRACTORS: Dict[str, object] = {}
_STATIC_EXTRACTORS: Dict[str, Type] = {
    "bca": BCAExtractor,
    "bni": BNIExtractor,
    "permata": PermataExtractor,
    "mandiri": MandiriExtractor,
}

def set_reader(reader) -> None:
    global _EASYOCR_READER
    _EASYOCR_READER = reader

def ensure_reader() -> None:
    """Lazy init untuk dev/CLI saat service dipanggil sebelum startup."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["id", "en"])

def path_for(name: str) -> Path:
    return BANKS_DIR / f"{name.lower()}.json"

def save_config(cfg: dict) -> Path:
    name = cfg.get("name")
    if not name:
        raise ValueError("Missing name in config")
    p = path_for(name)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def make_extractor_class(class_name: str, base_cls, config: dict):
    class DynamicExtractor(base_cls):
        def __init__(self, reader=None):
            try:
                super().__init__(reader)
            except TypeError:
                super(DynamicExtractor, self).__init__(reader)
            for k, v in config.items():
                if isinstance(k, str) and k.isidentifier():
                    setattr(self, k, v)
    DynamicExtractor.__name__ = class_name
    DynamicExtractor.__qualname__ = class_name
    return DynamicExtractor

def register_from_config(cfg: dict):
    ensure_reader()
    name = cfg.get("name")
    if not name:
        raise ValueError("Konfigurasi harus mengandung 'name'")
    key = name.lower()
    base_cls = DefaultBaseExtractor
    cls_name = f"{name}Extractor"
    Dyn = make_extractor_class(cls_name, base_cls, cfg)
    instance = Dyn(_EASYOCR_READER)
    _DYNAMIC_EXTRACTORS[key] = instance
    return instance

def load_all_on_startup():
    ensure_reader()
    for cfg_file in sorted(BANKS_DIR.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            register_from_config(cfg)
        except Exception as e:
            print(f"[banks] warning load {cfg_file.name}: {e}")

def current_extractors() -> Dict[str, object]:
    ensure_reader()
    static = {k: cls(_EASYOCR_READER) for k, cls in _STATIC_EXTRACTORS.items()}
    return {**static, **_DYNAMIC_EXTRACTORS}

def get_extractor(bank_name: str):
    ensure_reader()
    bank = bank_name.lower()
    ex = current_extractors().get(bank)
    if not ex:
        raise HTTPException(status_code=400, detail=f"Bank {bank_name} tidak terdaftar")
    if hasattr(ex, "reader") and ex.reader is None:
        ex.reader = _EASYOCR_READER
    return ex

def list_banks() -> list[str]:
    return sorted(set(list(_STATIC_EXTRACTORS.keys()) + list(_DYNAMIC_EXTRACTORS.keys())))

def delete_bank(name: str) -> bool:
    key = name.lower()
    p = path_for(key)
    if p.exists():
        p.unlink()
    removed_dyn = _DYNAMIC_EXTRACTORS.pop(key, None)
    removed_stat = _STATIC_EXTRACTORS.pop(key, None)
    return (removed_dyn is None and removed_stat is None)

def update_bank_files(old_key: str, old_cfg: dict, new_cfg: dict):
    old_path = path_for(old_key)
    new_key = (new_cfg.get("name") or old_key).lower()
    new_path = path_for(new_key)

    if new_key != old_key and new_path.exists():
        raise HTTPException(status_code=400, detail=f"Bank '{new_key}' sudah terdaftar")

    save_config(new_cfg)

    if new_key != old_key and old_path.exists():
        old_path.unlink(missing_ok=True)

    _DYNAMIC_EXTRACTORS.pop(old_key, None)
    try:
        register_from_config(new_cfg)
    except Exception as e:
        try:
            if new_key != old_key:
                new_path.unlink(missing_ok=True)
                old_path.write_text(json.dumps(old_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                path_for(old_key).write_text(json.dumps(old_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        try:
            register_from_config(old_cfg)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Gagal register extractor: {e}")
