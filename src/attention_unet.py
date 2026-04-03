# File: src/attention_unet.py
# EXACTLY matching Colab checkpoint shapes

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_g(gate)
        s = self.W_x(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:],
                              mode='bilinear', align_corners=True)
        attention = self.psi(self.relu(g + s))
        return skip * attention


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder with attention
        # CRITICAL: attention gate_ch = f (after upsample), NOT f*2
        for f in reversed(features):
            self.attentions.append(
                AttentionBlock(gate_ch=f, skip_ch=f, inter_ch=f // 2)
            )
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        att_idx = 0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)          # Upsample: now x has f channels
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=True)

            # Attention: gate=x (f channels), skip=skip (f channels)
            skip = self.attentions[att_idx](gate=x, skip=skip)
            att_idx += 1

            x = torch.cat([skip, x], dim=1)  # Now 2f channels
            x = self.ups[idx + 1](x)         # DoubleConv: 2f -> f

        return self.final_conv(x)


if __name__ == "__main__":
    model = AttentionUNet()
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")