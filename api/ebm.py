from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import os

router = APIRouter()

@router.get("/sample/ebm")
def sample_ebm():
    for path in ("ebm.png", "outputs/ebm_sample.png"):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="ebm.png")
    return JSONResponse({"ok": False, "note": "No EBM image found yet. After Langevin sampling, save a PNG to ebm.png."})
