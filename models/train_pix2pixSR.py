import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generatorSR import Generator
from discriminatorSR import Discriminator
from pix2pixSR import Pix2PixLoss


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
        if folder in["demo","jammu"]:
            continue
        subpath = os.path.join(root_dir, folder)

        for subfolder in sorted(os.listdir(subpath)):

            path = os.path.join(subpath, subfolder)

            if not os.path.isdir(path):
                continue

            tir = os.path.join(path, "tir_200m.npy")
            tir_hr = os.path.join(path, "rgb_100m_512.npy")

            if os.path.exists(tir) and os.path.exists(tir_hr):
                samples.append((tir, tir_hr))

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

        tir_path, tir_hr_path = self.samples[idx]

        tir = np.load(tir_path).astype(np.float32)
        tir_hr = np.load(tir_hr_path).astype(np.float32)

        tir = np.load(tir_path).astype(np.float32)

        # if 2D image
        if tir.ndim == 2:
            tir = tir

        # if 3D but has channel first
        elif tir.ndim == 3:
            tir = tir[0]

        tir = (tir - TIR_GLOBAL_MIN) / (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
        tir = np.clip(tir, 0.0, 1.0)

        # upsample
        tir = np.repeat(np.repeat(tir, 2, axis=0), 2, axis=1)
        


        tir_hr = (tir_hr - tir_hr.min()) / (tir_hr.max() - tir_hr.min() + 1e-6)
        tir_hr=(tir_hr[0]/3 + tir_hr[1]/3 + tir_hr[2]/3)
        tir_hr= tir_hr -tir_hr.mean() + tir.mean() 
        tir_hr= tir_hr*0.3 + tir*0.7
        tir_hr = np.clip(tir_hr, 0.0, 1.0)


        # -------------------------
        # CHANGED: 1-Channel Pure Super-Resolution Setup
        # -------------------------
        # Pehle 3 channels extract ho rahe the [tir, mean, grad]
        # Pure 1-channel target pipeline ke liye input sirf thermal single band (1, H, W) hona chahiye.
        physics = np.expand_dims(tir, axis=0) 

        # CHANGED: Agar target image 3 channels ki thi, toh use humne grayscale (1, H, W) me secure kar liya.
        if len(tir_hr.shape) == 3:
            if tir_hr.shape[-1] == 3:  # HWC to CHW format fix
                tir_hr = np.transpose(tir_hr, (2, 0, 1))
            # Grayscale transformation using standard luminance weights
            tir_hr = 0.299 * tir_hr[0:1, :, :] + 0.587 * tir_hr[1:2, :, :] + 0.114 * tir_hr[2:3, :, :]
        else:
            tir_hr = np.expand_dims(tir_hr, axis=0)

        return (
            torch.tensor(physics, dtype=torch.float32),
            torch.tensor(tir_hr, dtype=torch.float32)
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

# CHANGED: Explicitly models ko 1-channel configuration pass ki hai taaki pipeline sync ho jaye.
gen = Generator(in_channels=1, out_channels=1).to(DEVICE)
disc = Discriminator(in_channels=1, out_channels=1).to(DEVICE)

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

        for physics, tir_hr in val_loader:

            physics = physics.to(DEVICE)
            tir_hr = tir_hr.to(DEVICE)

            fake = gen(physics)

            total += loss_fn(fake, tir_hr).item()

    gen.train()

    return total / len(val_loader)


# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(EPOCHS):

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    

    for physics, tir_hr in loop:

        physics = physics.to(DEVICE)
        tir_hr = tir_hr.to(DEVICE)

        # ---------------------
        # GENERATOR
        # ---------------------
        fake = gen(physics)
        # print(fake.shape)
        # print("physics:", physics.shape)
        # print("tir_hr:", tir_hr.shape)

        # ---------------------
        # DISCRIMINATOR
        # ---------------------
        real_pred = disc(physics, tir_hr)
        fake_pred = disc(physics, fake.detach())

        d_loss = criterion.discriminator_loss(real_pred, fake_pred)

        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        # ---------------------
        # GENERATOR LOSS
        # ---------------------
        fake_pred = disc(physics, fake)

        # CHANGED: Pix2PixLoss me humne SSIM lagaya tha jo total 5 values unpack karta hai.
        # Pehle pichle script me sirf 4 variables unpack ho rahe the jis se error aata.
        g_loss, gan, l1, ssim_loss, phys = criterion.generator_loss(
            fake_pred,
            fake,
            tir_hr,
            temperature=physics[:, 0:1] # CHANGED: 1-channel index array dimension maintain ki hai.
        )

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        loop.set_postfix(
            D=f"{d_loss.item():.3f}",
            G=f"{g_loss.item():.3f}",
            SSIM=f"{ssim_loss.item():.3f}" # CHANGED: Loop progress bar me monitor karne ke liye add kiya hai.
        )

    # ---------------------
    # VALIDATION
    # ---------------------
    val_loss = validate()

    print(f"\n📊 Epoch {epoch+1} Validation L1: {val_loss:.4f}")

    # ---------------------
    # SAVE MODEL
    # ---------------------
    # CHANGED: Indentation fix, taaki model har epoch ke khatam hone ke baad hi save ho na ki har image patch par.
os.makedirs("checkpoints", exist_ok=True)
torch.save(gen.state_dict(), f"checkpoints/gen_SR_{epoch+1}.pth")
torch.save(disc.state_dict(), f"checkpoints/disc_SR_{epoch+1}.pth")


print("🚀 Training Complete")