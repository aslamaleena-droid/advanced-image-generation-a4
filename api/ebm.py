from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, JSONResponse
import os

router = APIRouter()

IMG_CANDIDATES = ("ebm.png", "outputs/ebm_sample.png")

@router.get("/sample/ebm")
def sample_ebm():
    for path in IMG_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="ebm.png")
    return JSONResponse({"ok": False, "note": "No EBM image found yet. After Langevin sampling, save a PNG to ebm.png."})

@router.head("/sample/ebm")
def head_ebm():
    exists = any(os.path.exists(p) and os.path.getsize(p) > 0 for p in IMG_CANDIDATES)
    headers = {"Content-Type": "image/png"} if exists else {}
    return Response(status_code=200, headers=headers)
