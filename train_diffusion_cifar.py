import os, math, torch, torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, utils
from models import TinyUNet
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = 'checkpoints/diffusion_cifar_unet.pth'
os.makedirs('checkpoints', exist_ok=True)

# --- data ---
tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # to [-1,1]
])
ds = datasets.CIFAR10(root='data', train=True, download=True, transform=tfm)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

# --- model ---
T = 200                       # small for speed
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    a_bar = alpha_bar[t].view(-1,1,1,1)
    return torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise

net = TinyUNet().to(device)
opt = optim.AdamW(net.parameters(), lr=1e-3)

epochs = 1  # small but enough to produce nontrivial samples with time
net.train()
for epoch in range(epochs):
    for i,(x,_ ) in enumerate(dl):
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device)
        eps = torch.randn_like(x)
        xt = q_sample(x, t, eps)
        pred = net(xt, t)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if i % 200 == 0:
            print(f"[{epoch}:{i}] loss={loss.item():.4f}")
            # early break to keep runtime short if desired
            # if i>400: break

torch.save({'model': net.state_dict(), 'T':T, 'alpha_bar':alpha_bar.cpu()}, ckpt_path)
print(f"✅ saved {ckpt_path}")

# --- quick sampler to write diff.png ---
@torch.no_grad()
def sample_png(n=16):
    net.eval()
    data = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(data['model'])
    T = data['T']; alpha_bar = data['alpha_bar'].to(device)
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas

    x = torch.randn(n,3,32,32, device=device)
    for t in reversed(range(T)):
        t_ = torch.full((n,), t, device=device, dtype=torch.long)
        eps = net(x, t_)
        a = alphas[t]; a_bar = alpha_bar[t]
        x = (x - (1-a)/torch.sqrt(1-a_bar)*eps) / torch.sqrt(a)
        if t>0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)
    x = (x.clamp(-1,1)+1)/2
    grid = utils.make_grid(x, nrow=int(math.sqrt(n)))
    utils.save_image(grid, "diff.png")
    print("🖼  wrote diff.png")

sample_png(16)
