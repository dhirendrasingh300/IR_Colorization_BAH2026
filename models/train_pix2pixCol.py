import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generatorcCol import Generator
from discriminatorCol import Discriminator
from pix2pixCol import Pix2PixLoss


# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = "output/patches"

BATCH_SIZE = 1   # IMPORTANT for 16 samples
EPOCHS = 100
LR = 2e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# =========================================================
# Global
# =========================================================

TIR_GLOBAL_MIN = 15000
TIR_GLOBAL_MAX = 35000





# =========================================================
# COLLECT PAIRS
# =========================================================

def collect_pairs(root_dir):

    samples = []
    for folder in sorted(os.listdir(root_dir)):
        if folder in ["demo","jammu"]:
            continue
        subpath= os.path.join(root_dir, folder)

        for subfolder in sorted(os.listdir(subpath)):

            path = os.path.join(subpath, subfolder)

            if not os.path.isdir(path):
                continue

            tir = os.path.join(path, "tir_100m_512.npy")
            rgb = os.path.join(path, "rgb_100m_512.npy")

            if os.path.exists(tir) and os.path.exists(rgb):
                samples.append((tir, rgb))

    return samples


# =========================================================
# SPLIT (16 samples optimized)
# =========================================================

def split_data(samples):

    random.shuffle(samples)

    n = len(samples)

    train_end = int(n * 0.75)   # 18
    val_end = train_end + 4     # 3 val

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    return train, val, test


# =========================================================
# DATASET
# =========================================================

class TIRDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        tir_path, rgb_path = self.samples[idx]

        tir = np.load(tir_path).astype(np.float32)
        rgb = np.load(rgb_path).astype(np.float32)

        # -------------------------
        # Normalize#############################################################3
        # -------------------------
        tir = (tir - TIR_GLOBAL_MIN )/ (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
        tir= np.clip(tir, 0.0, 1.0)

        rgb = (rgb-rgb.min()) / (rgb.max() -rgb.min() + 1e-6)
        rgb= np.clip(rgb, 0.0, 1.0)

        tir = np.squeeze(tir)

        # -------------------------
        # Physics Features
        # -------------------------
        mean = (
            np.roll(tir, 1, axis=0)
            + tir
            + np.roll(tir, -1, axis=0)
        ) / 3

        if tir.shape[0] < 2 or tir.shape[1] < 2:
            gx = np.zeros_like(tir)
            gy = np.zeros_like(tir)
        else:
            gy = np.gradient(tir, axis=0)
            gx = np.gradient(tir, axis=1)

        grad = np.sqrt(gx**2 + gy**2)
        grad = grad / (grad.max() + 1e-6)

        physics = np.stack([tir, mean, grad], axis=0)

        # RGB format fix
        if rgb.shape[-1] == 3:
            rgb = np.transpose(rgb, (2, 0, 1))

        return (
            torch.tensor(physics, dtype=torch.float32),
            torch.tensor(rgb, dtype=torch.float32)
        )


# =========================================================
# LOAD DATA
# =========================================================

all_samples = collect_pairs(DATA_ROOT)

train_s, val_s, test_s = split_data(all_samples)

train_loader = DataLoader(TIRDataset(train_s), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TIRDataset(val_s), batch_size=1)
test_loader = DataLoader(TIRDataset(test_s), batch_size=1)


# =========================================================
# MODEL
# =========================================================

gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

criterion = Pix2PixLoss()

opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))


# =========================================================
# VALIDATION
# =========================================================

def validate():

    gen.eval()
    loss_fn = nn.L1Loss()

    total = 0

    with torch.no_grad():

        for physics, rgb in val_loader:

            physics = physics.to(DEVICE)
            rgb = rgb.to(DEVICE)

            fake = gen(physics)

            total += loss_fn(fake, rgb).item()

    gen.train()

    return total / len(val_loader)


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(EPOCHS):

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for physics, rgb in loop:

        physics = physics.to(DEVICE)
        rgb = rgb.to(DEVICE)

        # ---------------------
        # GENERATOR
        # ---------------------
        fake = gen(physics)

        # ---------------------
        # DISCRIMINATOR
        # ---------------------
        real_pred = disc(physics, rgb)
        fake_pred = disc(physics, fake.detach())

        d_loss = criterion.discriminator_loss(real_pred, fake_pred)

        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        # ---------------------
        # GENERATOR LOSS
        # ---------------------
        fake_pred = disc(physics, fake)

        g_loss, gan, l1, edge = criterion.generator_loss(
            fake_pred,
            fake,
            rgb,
            temperature = physics[:, 0]
        )

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        loop.set_postfix(
            D=f"{d_loss.item():.3f}",
            G=f"{g_loss.item():.3f}"
        )

    # ---------------------
    # VALIDATION
    # ---------------------
    val_loss = validate()

    print(f"\n📊 Epoch {epoch+1} Validation L1: {val_loss:.4f}")

    # ---------------------
    # SAVE MODEL
    
# ---------------------
os.makedirs("checkpoints", exist_ok=True)

torch.save(gen.state_dict(), f"checkpoints/gen_{epoch+1}.pth")
torch.save(disc.state_dict(), f"checkpoints/disc_{epoch+1}.pth")


print("🚀 Training Complete")