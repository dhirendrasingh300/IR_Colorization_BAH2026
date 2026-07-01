# Infrared Image Colorization and Enhancement for Improved Object Interpretation


## 🚀 Bhartiya Antriksh Hackathon (BAH 2026)

---

## Team Name : NovapixelX4

### Problem Statement
### **Infrared Image Colorization and Enhancement for Improved Object Interpretation**

**Transforming Low-Resolution Thermal Infrared (TIR) Satellite Images into High-Resolution Thermal and Colorized RGB Images for Better Visual Understanding**

---

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-blue)
---


# 📌 Overview

Thermal Infrared (TIR) satellite imagery plays a crucial role in environmental monitoring, disaster management, wildfire detection, urban heat island analysis, volcanic monitoring, and night-time Earth observation. However, Thermal Infrared images are typically single-band grayscale images with lower native spatial resolution, making object interpretation difficult for human analysts.

This project presents a complete deep learning pipeline that enhances low-resolution Thermal Infrared imagery by first generating a **Super-Resolved Thermal Image** and then converting it into a visually interpretable **Colorized RGB Image**.

The proposed pipeline follows:

```
Low Resolution Thermal Image
            │
            ▼
High Resolution Thermal Image(Super Resolution)
            │
            ▼
Colorized TIR RGB Image
```

---

# 🎯 Objectives

The proposed solution aims to:

- Enhance low-resolution Thermal Infrared imagery
- Recover structural and spatial details using Super Resolution
- Generate realistic RGB images from Thermal Infrared imagery
- Improve visual interpretation of remote sensing imagery
- Preserve thermal information while enhancing image quality

---

# 🛰 Dataset

The project uses **Landsat-9** satellite imagery.

## Bands Used

| Band | Description |
|------|-------------|
| **B2** | Blue |
| **B3** | Green |
| **B4** | Red |
| **B10** | Thermal Infrared |

### Input Directory Structure

```
input/
├── <Location_1>/
│   ├── <file_prefix_1>_B10.TIF
│   ├── <file_prefix_1>_B2.TIF
│   ├── <file_prefix_1>_B3.TIF
│   └── <file_prefix_1>_B4.TIF
├── <Location_2>/
│   ├── <file_prefix_2>_B10.TIF
│   ├── <file_prefix_2>_B2.TIF
│   ├── <file_prefix_2>_B3.TIF
│   └── <file_prefix_2>_B4.TIF
└── <Location_3>/
    ├── <file_prefix_3>_B10.TIF
    ├── <file_prefix_3>_B2.TIF
    ├── ...
```
---
### In this project we used dta fron 3 locations(Delhi, Jammu & Lucknow)  
### From the above mentioned three locations we used two locations for training and one for testing. 
### To download the dataset you can use  [USGS Earth Explorer](https://earthexplorer.usgs.gov/); please ensure it is placed in the `input` directory following the structure below.
---

# 📊 Data Preparation

The preprocessing pipeline includes:

- Reading GeoTIFF satellite imagery
- Patch extraction
- Synthetic downsampling
- Percentile clipping(visualisation)
- Generation of paired datasets
- Band normalization(for each patches)

## Generated datasets:
```
output/
└── patches/
    ├── <Location_1>/
    │   ├── sample_000/
    │   │   ├── rgb_100m_512.npy
    │   │   ├── rgb_100m_512.png
    │   │   ├── tir_100m_512.npy
    │   │   ├── tir_100m_512.png
    │   │   ├── tir_200m.npy
    │   │   └── tir_200m.png
    │   ├── sample_001/
    │   │   └── ... (same 6 files)
    │   └── sample_015/
    │       └── ...
    └── <Location_2>/
        └── sample_000/
            └── ...
```
### Super Resolution

```
200m Thermal
        │
        ▼
100m Thermal
```

### Colorization

```
100m Thermal
        │
        ▼
100m RGB
```

Training patches are stored as **NumPy (.npy)** files to preserve the original radiometric information.

---

# 🧠 Proposed Methodology

The complete framework consists of two independent deep learning stages.

---
# Stage 1: Thermal Image Enhancement (Super Resolution)

## 1. Objective
The primary objective of this stage is to reconstruct a high-resolution, sharp Thermal Infrared (HR-TIR) image from a single-channel, low-resolution ($200\text{m}$) input. This enhancement process is critical for capturing fine-grained structural thermal boundaries and subtle gradients that are typically lost due to the hardware limitations of low-resolution satellite sensors.

---

## 2. Pipeline Overview

```
                  +-------------------------------+
                  |      Input Low-Res TIR        |
                  |     (200m Spatial Res)        |
                  |     Size: 1 x 256 x 256       |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |  Preprocessing: Interpolation |
                  |        (Scale-Up x2)          |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |    Spatially Upsampled TIR    |
                  |   (Interpolated / Blurry)     |
                  |     Size: 1 x 512 x 512       |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |   Pix2Pix Generator (U-Net)   |
                  |  (Learns texture/detail map)  |
                  +---------------+---------------+
                                  |
                                  v
                  +---------------+---------------+
                  |      Output High-Res TIR      |
                  |     (100m Spatial Res)        |
                  |     Size: 1 x 512 x 512       |
                  +-------------------------------+
```


## 3. Model Architecture: Pix2Pix (cGAN)
Instead of traditional mathematical upsamplers that rely purely on localized pixel interpolation, this stage employs a customized **Pix2Pix (Conditional GAN)** framework. The model learns a direct mapping from the blurred upscaled matrix to a crisply defined high-resolution target structure.

```
       [Input Spatially Scaled TIR] (Condition)
                    │
                    ▼
              ┌───────────┐
              │ Generator │ ──► [Generated HR-TIR] ──┐
              │  (U-Net)  │                          │
              └───────────┘                          ▼
                                               ┌───────────────┐
                [Real HR-TIR] (Ground Truth) ─►│ Discriminator │ ──► [Real / Fake]
                                               │  (PatchGAN)   │
       [Input Spatially Scaled TIR] (Condition) ─►└───────────────┘
```

## 4. Advantages
- Lightweight architecture
- Fast inference
- Efficient Pixel Shuffle Upsampling
- Better reconstruction quality
- CPU friendly

---

# Stage 2 : Thermal Image Colorization
# Pix2Pix Training Pipeline Documentation

This document provides a concise overview of the custom Pix2Pix architecture implemented in the script. The model translation task translates Thermal Infrared (TIR) physics-enhanced input into high-fidelity RGB imagery.

---

## Pipeline Architecture

The pipeline processes input NumPy matrices, extracts handcrafted physical features, applies conditional generative adversarial networks (cGAN), and validates performance using cross-entropy and geometric reconstruction loss formulations.

```
+---------------------------+
| TIR 100m (512x512) Matrix  |
+-------------+-------------+
              |
              v (Physics Extraction)
+-------------------------------------------+
| 3-Channel Input Tensor                    |
| 1. Normalized TIR (Min-Max Global)        |
| 2. Spatial Rolling Local Mean (3x3 Blur)  |
| 3. High-Pass Spatial Gradient Magnitude   |
+---------------------+---------------------+
                      |
                      v
          +-----------+-----------+
          |   Pix2Pix Generator   |
          +-----------+-----------+
                      |
                      v
          +-----------+-----------+
          |  Generated colorised  | 
          |         TIR RGB       |
          +-----------+-----------+
```

---

## Data Preprocessing & Physics Injection

Instead of passing the raw Single-Channel Thermal Infrared image (`tir_100m_512.npy`), the dataset injects localized domain physics to assist spatial texturing:

1. **Global Normalization**: Bounded using pre-calibrated limits (`TIR_GLOBAL_MIN = 15000`, `TIR_GLOBAL_MAX = 35000`) to maintain temporal radiative balance.
2. **Local Smoothed Mean**: Spatial convolution mimicking thermal diffusion across adjacent pixels (`np.roll` along spatial dimensions).
3. **Thermal Gradient Magnitude**: Derived via dynamic spatial 2D gradients ($G = \sqrt{g_x^2 + g_y^2}$) highlighting structural boundaries, fault lines, or material interfaces.

---

## Loss Configuration & Backpropagation

The architecture runs a conditioned optimization sequence where the Discriminator checks structural correctness against both the true input terrain context and the target RGB profile.

### 1. Discriminator Update
Optimizes the ability to distinguish true pairs from generated synthesis:
$$\mathcal{L}_{D} = 	ext{BCE}(\mathcal{D}(	ext{Physics}, 	ext{RGB}), 1) + 	ext{BCE}(\mathcal{D}(	ext{Physics}, 	ext{Gen}(	ext{Physics})), 0)$$

### 2. Generator Update
Optimizes a multi-objective composite loss comprising adversarial cheat rate, physical parameter temperature weights, reconstruction $L_1$ distances, and edge coherence alignments:
$$\mathcal{L}_{G} =  lpha \cdot \mathcal{L}_{	ext{GAN}} +  eta \cdot \mathcal{L}_{1} + \gamma \cdot \mathcal{L}_{	ext{edge}}$$

---

## Hyperparameter Configuration Reference

| Parameter | Configuration Value | Operational Significance |
| :--- | :--- | :--- |
| **Batch Size** | `1` | Strictly forced due to pixel-to-pixel structural dependency checks |
| **Learning Rate** | `2e-4` | Balanced standard step size optimized for Adam optimizer |
| **Betas** | `(0.5, 0.999)` | Decreased $ eta_1$ momentum prevents sudden training collapses |
| **Epoch Run** | `100` | Full operational sweep over structural spatial targets |
| **Device Execution** | `cuda` dynamic fall-back | Automates hardware-accelerated tensor matrix pipelines |

---

## Model Checkpoint Targets
The script automatically sets up export matrices in sequential runtime stages: you can download the checkpoints through the following [Google Drive Link]( https://drive.google.com/drive/folders/1XpfI3YZ3zdv0tCkZO5nL2Lqyts0nHeoX?usp=drive_link )
- **Weights directory**: `/checkpoints/`
- **Output naming schema**: `gen_100.pth` & `disc_100.pth`
## Objective

Generate realistic RGB imagery from the Super-Resolved Thermal image.

### Model

**Pix2Pix Conditional GAN (cGAN)**

### Generator

- U-Net Architecture
- Skip Connections
- Encoder-Decoder Network

### Discriminator

- PatchGAN

### Training Loss

- Adversarial Loss
- L1 Reconstruction Loss


---

# 🔄 Complete Pipeline

```
        Input Thermal Infrared Image (200m)
                        │
                        ▼
            Thermal Image Normalization
                        │
                        ▼
              ESPCN Super Resolution
                        │
                        ▼
        High Resolution Thermal Image (100m)
                        │
                        ▼
  New feature Extraction (1,512,512)-> (3,512,512)
                        │
                        ▼
              Pix2Pix Conditional GAN
                        │
                        ▼
             Colorized RGB Image (100m)
                        │
                        ▼
              GeoTIFF Export (png/npy)
```

---

# 📂 Project Structure
### To download the checkpoint file you can use the following [Google Drive Link ]( https://drive.google.com/drive/folders/1XpfI3YZ3zdv0tCkZO5nL2Lqyts0nHeoX?usp=drive_link)
```
IR-COLORIZATION-BAH2026-MAIN/
├── checkpoints/                        # Model weights (.pth) for both stages
│   ├── disc_100.pth                    
│   ├── disc_SR_100.pth                
│   ├── gen_100.pth                     
│   └── gen_SR_100.pth                  
├── input/                              # Raw geospatial multi-band inputs B2,B3,B4,B10
│   ├── delhi/                          # Delhi Landsat-9 band folders
│   ├── jammu/                          # Jammu Landsat-9 band folders
│   └── lucknow/                        # Lucknow Landsat-9 band folders
├── models/                             # Core neural network architectures and trainers
│   ├── discriminatorCol.py             
│   ├── discriminatorSR.py              
│   ├── generatorCol.py                
│   ├── generatorSR.py                
│   ├── pix2pixCol.py                   
│   ├── pix2pixSR.py                   
│   ├── Run_code.py                     
│   ├── testCol.py                      
│   ├── testSR.py                      
│   ├── train_pix2pixCol.py             
│   └── train_pix2pixSR.py             
├── output/                             
│   ├── downscaled_data/                
│   ├── patches/                        # Extracted  patches
│   ├── rgb_images/                     
│   └── patches.log                     
├── results/                            # Final evaluation outputs (Visualizations & Arrays)
│   ├── RGB.npy                         
│   ├── RGB.png                         
│   ├── SR.npy                         
│   ├── SR.png                          
│   ├── tir.npy                        
│   └── tir.png                        
├── scripts/                            # Dataset engineering automation
│   ├── __init__.py
│   ├── create_patches.py               
│   ├── download_data.sh                
│   ├── download.py                     
│   ├── downscale.py                   
│   └── merge_rgb.py                    
├── utils/                              # System helpers and auxiliary scripts
│   ├── file_utils.py                   
│   ├── logging_utils.py               
│   └── visualization.py                
├── driver.py                           # Master driver for preprocessing orchestration
├── eval.py                             # Testing and metric reporting suite(MSE, MAE, PSNR, SSIM)
├── output.log                          # Comprehensive runtime logs
├── README_PROBLEM.md                   # Main Hackathon documentation (Version 1)
├── README_GUIDE.md                     # Main Hackathon documentation (Version 3)
├── Run_model.py                        # Top-level entry point for inference execution
├── tempCodeRunnerFile.py               # Cache environment running file
└── NovaPixelX4.pdf                     # PDF Report

```

---

# ⚙ Training
# Installation

Install all the required Python libraries using pip:

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib tifffile tqdm scikit-image
```

## Included Libraries

- torch
- torchvision
- numpy
- opencv-python
- Pillow
- matplotlib
- tifffile
- tqdm
- scikit-image

## Built-in Python Modules (No Installation Required)

The following modules are part of Python's standard library and do **not** require installation:

- os
- glob
- random
- argparse
- logging

### Generate Dataset

```bash
python models/driver.py
```

### Train Super Resolution Model

```bash
python models/train_pix2pixSR.py
```

### Train Colorization Model

```bash
python models/train_pix2pixCol.py
```

### Model Evaluation

```bash
python eval.py
```

---

### Testing sample

```bash
python Run_model.py
```

Generated outputs are saved as(result)

```
├── results/
│   ├── RGB.npy
│   ├── RGB.png
│   ├── SR.npy
│   ├── SR.png
│   ├── tir.npy
│   ├── tir.png

```

---

# 📈 Sample Results

```
        Raw Thermal Image(saved in result\(tir.npy/.png))
                                │
                                ▼
     Super-Resolved Thermal Image(saved in result\(SR.npy/.png))
                                │
                                ▼
        Colorized RGB Image(saved in result\(RGB.npy/.png))
```

| Input TIR | Super-Resolved TIR | Colorized RGB |
|-----------|--------------------|---------------|
| ✅ | ✅ | ✅ |

The generated RGB images preserve thermal structures while significantly improving visual interpretation and object understanding.

---

# 📊 Evaluation Metrics

Performance is evaluated using:
- Images in (jammu): 16   
- Average MSE      :  0.03563906056660926   
- Average MAE      :  0.13452064199373126  
- Average PSNR     : 18.564363163687815  
- Average SSIM     : 0.50823295

 Mean Squared Error (MSE)   
 Mean Absolute Error (MAE)
 PSNR (Peak Signal-to-Noise Ratio)   
 SSIM (Structural Similarity Index)  

---

# ✨ Key Features

- End-to-End Deep Learning Pipeline
- Thermal Image Super Resolution
- Thermal to RGB Colorization
- Pix2Pix Conditional GAN
- Flaxible Output Support
- Radiometric Information Preservation
- Modular Code Structure
- Easy to Train and Extend

---

# 🌍 Applications

- Wildfire Detection
- Disaster Monitoring
- Environmental Monitoring
- Urban Heat Island Analysis
- Flood Assessment
- Forest Fire Monitoring
- Agricultural Monitoring
- Defence & Surveillance
- Night-Time Earth Observation
- Remote Sensing Research

---

# 💻 Technology Stack

- Python
- PyTorch
- OpenCV
- Tifffile
- NumPy
- Matplotlib


---

# 🔬 Future Work

Future improvements include:

- SwinIR-based Thermal Super Resolution
- Transformer-based Colorization
- Attention Mechanisms
- Physics-aware Loss Functions
- Multi-Spectral Feature Fusion
- Self-Supervised Thermal Pretraining
- Diffusion-based Image Colorization
- Cloud Deployment for Large-Scale Inference

---

# 👨‍💻 Team

**Bhartiya Antriksh Hackathon (BAH) 2026  
Team : NovapixelX4  
Team Members:**  
Dhirendra Kumar Singh (Central University of Jammu)  
Devishree Maddu (Central University of Jammu)   
Bhargavi Choudhary (Central University of Jammu)   
Divyanshu Tiwari (Central University of Jammu) 


---

# 📜 License

This project has been developed exclusively for academic, research, and innovation purposes as part of Bhartiya Antriksh Hackathon (BAH) 2026.



