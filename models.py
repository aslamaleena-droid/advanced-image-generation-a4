import math, torch, torch.nn as nn, torch.nn.functional as F

# ------- sinusoidal time embedding -------
def sinusoidal_embed(t, dim=64, max_period=10000.0):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ------- tiny UNet for 32x32 CIFAR -------
class ResBlock(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.emb = nn.Linear(tdim, c)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, tdim); self.down1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.rb2 = ResBlock(base*2, tdim); self.down2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.rb3 = ResBlock(base*4, tdim)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1); self.rb4 = ResBlock(base*2, tdim)
        self.up2 = nn.ConvTranspose2d(base*2, base, 4, 2, 1); self.rb5 = ResBlock(base, tdim)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        temb = sinusoidal_embed(t, self.tdim)
        h = self.inp(x)
        h1 = self.rb1(h, temb); h2 = self.down1(h1)
        h2 = self.rb2(h2, temb); h3 = self.down2(h2)
        h3 = self.rb3(h3, temb)
        h = self.up1(h3); h = self.rb4(h, temb)
        h = self.up2(h);  h = self.rb5(h, temb)
        return self.out(h)

# ------- tiny EBM (energy scalar) -------
class EBMNet(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(),   # 32->16
            nn.Conv2d(64,128,3, padding=1), nn.SiLU(),
            nn.Conv2d(128,128,4, 2, 1), nn.SiLU(),  # 16->8
            nn.Conv2d(128,256,3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256,1,1)
        )
    def forward(self, x):
        return self.net(x).flatten(1).squeeze(1)
