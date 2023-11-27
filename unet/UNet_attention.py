import math
import torch
import torch.nn as nn

from unet.UNet import sinusoidal_embedding, MyBlock


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(16, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x
    

class MyUNetattention(nn.Module):
  # Here is a network with 3 down and 3 up with the tiny block adapted to cifar 10 dataset with the image size "32*32"
    def __init__(self, shape, n_steps=1000, time_emb_dim=100):
        super().__init__()

        # Sinusoidal embedding
        H, W = shape
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((3, H, W), 3, 16),
            MyBlock((16, H, W), 16, 16),
            MyBlock((16, H, W), 16, 16)
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = nn.Sequential(
            MyBlock((16, H//2, W//2), 16, 32),
            MyBlock((32, H//2, W//2), 32, 32),
            MyBlock((32, H//2, W//2), 32, 32)
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = nn.Sequential(
            MyBlock((32, H//4, W//4), 32, 64),
            MyBlock((64, H//4, W//4), 64, 64),
            MyBlock((64, H//4, W//4), 64, 64)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 64)
        self.b_mid = nn.Sequential(
            MyBlock((64, H//8, W//8), 64, 32),
            MyBlock((32, H//8, W//8), 32, 32),
            AttentionBlock(32, num_heads=2),
            MyBlock((32, H//8, W//8), 32, 64)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 128)
        self.b4 = nn.Sequential(
            MyBlock((128, H//4, W//4), 128, 64),
            MyBlock((64, H//4, W//4), 64, 32),
            MyBlock((32, H//4, W//4), 32, 32)
        )

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 64)
        self.b5 = nn.Sequential(
            MyBlock((64, H//2, W//2), 64, 32),
            MyBlock((32, H//2, W//2), 32, 16),
            MyBlock((16, H//2, W//2), 16, 16)
        )

        self.up3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = nn.Sequential(
            MyBlock((32, H, W), 32, 16),
            MyBlock((16, H, W), 16, 16),
            MyBlock((16, H, W), 16, 16, normalize=False)
        )

        self.conv_out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x, t):
 
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))