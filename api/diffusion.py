from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, JSONResponse
import os, torch
from torchvision import utils
from models import TinyUNet
from PIL import Image

router = APIRouter()
IMG_CANDIDATES = ("diff.png", "outputs/diffusion_sample.png")

def generate_if_needed():
    if any(os.path.exists(p) and os.path.getsize(p)>0 for p in IMG_CANDIDATES):
        return
    ckpt = "checkpoints/diffusion_cifar_unet.pth"
    if not os.path.exists(ckpt): return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(ckpt, map_location=device)
    net = TinyUNet().to(device); net.load_state_dict(data['model']); net.eval()
    T = data['T']; alpha_bar = data['alpha_bar'].to(device)
    betas = torch.linspace(1e-4, 0.02, T).to(device); alphas = 1.0 - betas
    x = torch.randn(16,3,32,32, device=device)
    with torch.no_grad():
        for t in reversed(range(T)):
            t_ = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps = net(x, t_)
            a = alphas[t]; a_bar = alpha_bar[t]
            x = (x - (1-a)/torch.sqrt(1-a_bar)*eps) / torch.sqrt(a)
            if t>0: x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
        x = (x.clamp(-1,1)+1)/2
        utils.save_image(utils.make_grid(x, nrow=4), "diff.png")

@router.get("/sample/diffusion")
def sample_diffusion():
    generate_if_needed()
    for path in IMG_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return FileResponse(path, media_type="image/png", filename="diffusion.png")
    return JSONResponse({"ok": False, "note": "No diffusion image or checkpoint found."})

@router.head("/sample/diffusion")
def head_diffusion():
    exists = any(os.path.exists(p) and os.path.getsize(p) > 0 for p in IMG_CANDIDATES)
    headers = {"Content-Type": "image/png"} if exists else {}
    return Response(status_code=200, headers=headers)
