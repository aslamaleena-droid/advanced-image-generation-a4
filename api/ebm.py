from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, JSONResponse
import os, torch
from torchvision import utils
from models import EBMNet

router = APIRouter()
IMG_CANDIDATES = ("ebm.png", "outputs/ebm_sample.png")

def generate_if_needed():
    if any(os.path.exists(p) and os.path.getsize(p)>0 for p in IMG_CANDIDATES):
        return
    ckpt = "checkpoints/ebm_cifar.pth"
    if not os.path.exists(ckpt): return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = EBMNet().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)['model'])
    net.eval()
    x = torch.randn(16,3,32,32, device=device); x.requires_grad_(True)
    S, a, sig = 200, 0.05, 0.01
    for _ in range(S):
        en = net(x).sum()
        grad = torch.autograd.grad(en, x, retain_graph=False)[0]
        x = (x - a*grad + sig*torch.randn_like(x)).detach()
        x.requires_grad_(True)
    x = (x.clamp(-1,1)+1)/2
    utils.save_image(utils.make_grid(x, nrow=4), "ebm.png")

@router.get("/sample/ebm")
def sample_ebm():
    generate_if_needed()
    for path in IMG_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="ebm.png")
    return JSONResponse({"ok": False, "note": "No EBM image or checkpoint found."})

@router.head("/sample/ebm")
def head_ebm():
    exists = any(os.path.exists(p) and os.path.getsize(p) > 0 for p in IMG_CANDIDATES)
    headers = {"Content-Type": "image/png"} if exists else {}
    return Response(status_code=200, headers=headers)
