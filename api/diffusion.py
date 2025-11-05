from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import os

router = APIRouter()

@router.get("/sample/diffusion")
def sample_diffusion():
    # Serve an image file if present (drop a generated CIFAR sample here later)
    for path in ("diff.png", "outputs/diffusion_sample.png"):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="diffusion.png")
    return JSONResponse({"ok": False, "note": "No diffusion image found yet. Train or export a PNG to project root as diff.png."})
