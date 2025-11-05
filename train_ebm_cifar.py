import os, torch, torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, utils
from models import EBMNet
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = 'checkpoints/ebm_cifar.pth'
os.makedirs('checkpoints', exist_ok=True)

# --- data ---
tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
ds = datasets.CIFAR10(root='data', train=True, download=True, transform=tfm)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

net = EBMNet().to(device)
opt = optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)

# Langevin hyperparams
steps = 40
step_size = 0.1
noise = 0.01

epochs = 1  # short for demo
net.train()
for epoch in range(epochs):
    for i,(x,_ ) in enumerate(dl):
        x = x.to(device)

        # negative samples via Langevin starting from noise
        x_neg = torch.randn_like(x).to(device).detach()
        x_neg.requires_grad_(True)
        for _ in range(steps):
            en = net(x_neg).sum()
            grad = torch.autograd.grad(en, x_neg, create_graph=False)[0]
            x_neg = (x_neg - step_size * grad + noise * torch.randn_like(x_neg)).detach()
            x_neg.requires_grad_(True)

        # energy expectations: push down on data, up on negatives
        loss = net(x).mean() - net(x_neg.detach()).mean()

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if i % 200 == 0:
            print(f"[{epoch}:{i}] EBM loss={loss.item():.4f}")

torch.save({'model': net.state_dict()}, ckpt_path)
print(f"✅ saved {ckpt_path}")

# --- simple sampler to write ebm.png using Langevin from noise ---
@torch.no_grad()
def save_samples(n=16):
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    x = torch.randn(n,3,32,32, device=device)
    x.requires_grad_(True)
    S, a, sig = 200, 0.05, 0.01  # more steps for nicer images
    for _ in range(S):
        en = net(x).sum()
        grad = torch.autograd.grad(en, x, retain_graph=False)[0]
        x = (x - a*grad + sig*torch.randn_like(x)).detach()
        x.requires_grad_(True)
    x = (x.clamp(-1,1)+1)/2
    grid = utils.make_grid(x, nrow=4)
    utils.save_image(grid, "ebm.png")
    print("🖼  wrote ebm.png")

save_samples(16)
