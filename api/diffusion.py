from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, JSONResponse
import os

router = APIRouter()

IMG_CANDIDATES = ("diff.png", "outputs/diffusion_sample.png")

@router.get("/sample/diffusion")
def sample_diffusion():
    for path in IMG_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="diffusion.png")
    return JSONResponse({"ok": False, "note": "No diffusion image found yet. Train or export a PNG to project root as diff.png."})

@router.head("/sample/diffusion")
def head_diffusion():
    exists = any(os.path.exists(p) and os.path.getsize(p) > 0 for p in IMG_CANDIDATES)
    headers = {"Content-Type": "image/png"} if exists else {}
    return Response(status_code=200, headers=headers)
