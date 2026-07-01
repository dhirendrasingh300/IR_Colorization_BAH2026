"""
=========================================================
generator.py

Physics-Informed Pix2Pix Generator (Single Channel Input)
---------------------------------------------------------

Input Channels:
    Channel 1 -> B10 Thermal Band (or any Single Band)

Input Shape:
    (B, 1, 512, 512)

Output:
    RGB Image
    (B, 3, 512, 512)

Architecture:
    U-Net Generator with Skip Connections

Author:
    Bhartiya Antriksh Hackathon 2026
=========================================================
"""

import torch
import torch.nn as nn


# =========================================================
# Encoder / Decoder Block
# =========================================================

class Block(nn.Module):
    """
    Generic U-Net block used for:
        - Encoder (downsampling)
        - Decoder (upsampling)
    """

    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()

        if down:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


# =========================================================
# U-Net Generator
# =========================================================

class Generator(nn.Module):
    """
    Pix2Pix U-Net Generator configured for 1-channel Input and 3-channel Output.

    Input:
        (B, 1, 512, 512)

    Output:
        (B, 3, 512, 512)
    """

    def __init__(self, in_channels=1, out_channels=3):  # CHNAGES: in_channels set to 1
        super().__init__()

        # =========================
        # Encoder
        # =========================

        # No BatchNorm on the first layer to preserve raw input features
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = Block(64, 128, down=True)
        self.down3 = Block(128, 256, down=True)
        self.down4 = Block(256, 512, down=True)
        self.down5 = Block(512, 512, down=True)
        self.down6 = Block(512, 512, down=True)
        self.down7 = Block(512, 512, down=True)

        # =========================
        # Bottleneck
        # =========================

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)  # Tip: LeakyReLU works better for deep bottlenecks
        )

        # =========================
        # Decoder
        # =========================

        self.up1 = Block(512, 512, down=False, use_dropout=True)
        self.up2 = Block(1024, 512, down=False, use_dropout=True)
        self.up3 = Block(1024, 512, down=False, use_dropout=True)
        self.up4 = Block(1024, 512, down=False)
        self.up5 = Block(1024, 256, down=False)
        self.up6 = Block(512, 128, down=False)
        self.up7 = Block(256, 64, down=False)

        # =========================
        # Final Output Layer
        # =========================

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            # Target image pixel value limits ke hisab se activation switch karein:
            nn.Tanh()      # Use if targets are normalized to [-1, 1] (Standard Pix2Pix)
            # nn.Sigmoid() # Use if targets are normalized to [0, 1]
        )

    def forward(self, x):

        # =========================
        # Encoder
        # =========================

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # =========================
        # Bottleneck
        # =========================

        bottleneck = self.bottleneck(d7)

        # =========================
        # Decoder + Skip Connections
        # =========================

        u1 = self.up1(bottleneck)

        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        output = self.final(torch.cat([u7, d1], dim=1))

        return output


# =========================================================
# Quick Test
# =========================================================

if __name__ == "__main__":

    # Default is now 1 channel input
    model = Generator(in_channels=1, out_channels=3)

    # Creating a single-channel input tensor (B, C, H, W) -> (1, 1, 512, 512)
    x = torch.randn(1, 1, 512, 512)
    y = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)