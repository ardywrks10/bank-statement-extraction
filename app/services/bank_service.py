from typing import Dict, List
from fastapi import HTTPException
from app.dto.bank_dto import BankConfigIn, BankConfigUpdate
from app.extractors import dynamic_registry as reg

class BankService:
    def create(self, cfg_in: BankConfigIn) -> Dict:
        key = cfg_in.name.lower()
        path = reg.path_for(key)
        # TOLAK jika nama sudah ada (static/dynamic) atau file sudah ada
        if key in reg.current_extractors().keys() or path.exists():
            raise HTTPException(status_code=400, detail=f"Bank '{cfg_in.name}' sudah terdaftar")

        cfg = cfg_in.model_dump()
        reg.save_config(cfg)
        reg.register_from_config(cfg)
        return {"message": f"Bank '{cfg_in.name}' berhasil dibuat", "bank_key": key}

    def list(self) -> Dict[str, List[str]]:
        return {"banks": reg.list_banks()}

    def delete(self, name: str) -> Dict:
        key = name.lower()
        if (reg.path_for(key).exists() is False) and (key not in reg.current_extractors().keys()):
            raise HTTPException(status_code=404, detail="Bank tidak ditemukan")
        unreg = reg.delete_bank(key)
        return {"message": f"Bank '{name}' dihapus", "unregistered": unreg}

    def update(self, bank_name: str, patch: BankConfigUpdate) -> Dict:
        old_key = bank_name.lower()
        p = reg.path_for(old_key)
        if not p.exists():
            raise HTTPException(status_code=404, detail="Bank tidak ditemukan")

        try:
            old_cfg_text = p.read_text(encoding="utf-8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal membaca konfigurasi lama: {e}")

        import json
        old_cfg = json.loads(old_cfg_text)

        # Build patch dict (field yang diisi saja)
        patch_dict = {}
        for f in ["name","HEADERS","keterangan","kolom_kode","target_kode","debit_code","kredit_code","DATE_FORMAT","header_per_page"]:
            v = getattr(patch, f)
            if v is not None:
                patch_dict[f] = v

        new_cfg = dict(old_cfg)
        new_cfg.update(patch_dict)
        if "name" not in new_cfg or not new_cfg["name"]:
            new_cfg["name"] = old_cfg.get("name") or old_key

        reg.update_bank_files(old_key, old_cfg, new_cfg)
        changed_fields = list(patch_dict.keys())

        return {
            "message": f"Bank '{old_key}' diperbarui",
            "bank_key": (new_cfg.get("name") or old_key).lower(),
            "changed_fields": changed_fields,
            "before": old_cfg,
            "after": new_cfg,
        }
