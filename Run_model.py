import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from models.generatorSR import Generator as GeneratorSR
from models.generatorcCol import Generator as GeneratorCol
from utils.visualization import percentile_stretch


print("✅ import done ")



# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH_SR = "checkpoints/gen_SR_100.pth"
CHECKPOINT_PATH = "checkpoints/gen_100.pth"
save_path = "results"
output_rgb_path="results"

os.makedirs(save_path, exist_ok=True)


# =========================================================
# LOAD MODEL
# =========================================================


TIR_GLOBAL_MIN = 15000
TIR_GLOBAL_MAX = 35000

RGB_GLOBAL_MIN = 7000
RGB_GLOBAL_MAX = 25000


# =========================================================
# LOAD MODEL
# =========================================================

def load_modelSR():
    model = GeneratorSR(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH_SR, map_location=DEVICE))
    model.eval()
    return model
def load_modelCol():
    model = GeneratorCol().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

# =========================================================
# LOAD SAMPLE
# =========================================================

def load_sample(tir_path):

    tir = np.load(tir_path).astype(np.float32)
    if tir.ndim == 2:
        tir = tir
    
    elif tir.ndim == 3:
        tir = tir[0]
    tir = (tir - TIR_GLOBAL_MIN) / (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
    tir = np.clip(tir, 0.0, 1.0)

    # upsample
    tir = np.repeat(np.repeat(tir, 2, axis=0), 2, axis=1)
    physics = np.expand_dims(tir, axis=0) 
    

    return physics



def load_sample2(tir,rgb_path=None):

    # ---------------- TIR ----------------
    # tir = np.load(tir_path).astype(np.float32)
    tir = np.squeeze(tir)

    tir = (tir - TIR_GLOBAL_MIN )/ (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
    tir= np.clip(tir, 0.0, 1.0)

    mean = (np.roll(tir, 1, 0) + tir + np.roll(tir, -1, 0)) / 3

    gx = np.gradient(tir, axis=1)
    gy = np.gradient(tir, axis=0)

    grad = np.sqrt(gx**2 + gy**2)
    grad = grad / (grad.max() + 1e-6)

    physics = np.stack([tir, mean, grad], axis=0)
    physics = torch.tensor(physics, dtype=torch.float32).unsqueeze(0)

    # ---------------- RGB ----------------
    rgb = None
    if rgb_path:
        rgb = np.load(rgb_path).astype(np.float32)

        if rgb.shape[-1] == 3:
            rgb = np.transpose(rgb, (2, 0, 1))

        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        # rgb = (rgb ) / (rgb.max() + 1e-6)

        rgb = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)

    return physics, rgb


    return physics




# =========================================================
# INFERENCE
# =========================================================

def predictSR(model, physics):
    with torch.no_grad():

        # convert numpy → tensor
        physics = torch.tensor(physics, dtype=torch.float32)

        if physics.dim() == 2:
            physics = physics.unsqueeze(0).unsqueeze(0)   # H,W → 1,1,H,W

        elif physics.dim() == 3:
            physics = physics.unsqueeze(0)  # C,H,W → 1,C,H,W

        physics = physics.to(DEVICE)

        out = model(physics)

    return out.cpu().squeeze(0).numpy()



def predictCol(model, physics):
    with torch.no_grad():
        out = model(physics.to(DEVICE))
    return out.cpu().squeeze(0).numpy()


# =========================================================
# CLEAN (CHW → HWC)
# =========================================================

def clean(img):

    if img is None:
        return None

    img = np.array(img)

    if img.ndim == 4:
        img = img[0]

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    return img


# =========================================================
# CONVERT TO PNG (IMPORTANT PART)
# =========================================================

def save_png(tir_SR, save_path,name):
    """
    Save normalized RGB image as PNG.

    Parameters
    ----------
    tir_SR : numpy.ndarray
        Shape: (3,H,W) or (H,W,3)
        Range: [0,1]
    save_path : str
        Example: "results/fake.png"
    """
    save_path1=save_path +"/"+ name+".png"
    save_path2=save_path +"/"+ name+".npy"
    save_path = save_path1

    img = np.array(tir_SR)

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # Clip to valid range
    img = np.clip(img, 0.0, 1.0)

    # Convert to uint8
    img = (img * 255).round().astype(np.uint8)

    # RGB -> BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, img)
    np.save(save_path2, np.transpose(img, (2, 0, 1)))# HWC -> CHW 

    print(f"✅ PNG saved: {save_path}")


# =========================================================
# VISUALIZATION
# =========================================================

def visualize_numpy(tir, tir_SR=None, real_rgb=None):

    def prepare(img):
        if img is None:
            return None

        # Remove batch dimension
        if img.ndim == 4:
            img = img[0]

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        # Remove grayscale channel
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        # Normalize if values are in [-1,1]
        if img.max() > 1.5:
            # img = (img + 1) / 2
            # img = (img-RGB_GLOBAL_MIN) / (RGB_GLOBAL_MAX-RGB_GLOBAL_MIN + 1e-6)
            img = (img-img.min()) / (img.max() - img.min() + 1e-6)
    
          
            # img = img / img.max()

        img = np.clip(img, 0, 1)

        return img

    tir = prepare(tir)
    real_rgb = prepare(real_rgb)
    tir_SR = prepare(tir_SR)


    save_png(tir_SR, save_path,"SR")
    save_png(real_rgb, save_path,"RGB")
    save_png(tir, save_path,"tir")

    print(tir_SR.shape,type(tir_SR))
    print(tir_img.shape,type(tir_img))
    real_rgb=clean(real_rgb)
    print(real_rgb.shape,type(real_rgb))

    cols = 1 + (real_rgb is not None) + (tir_SR is not None)

    plt.figure(figsize=(5 * cols, 5))

    i = 1
    plt.subplot(1, cols, i)
    plt.imshow(tir, cmap='gray')
    plt.title("Input TIR")
    plt.axis("off")
    i += 1
    if tir_SR is not None:
        plt.subplot(1, cols, i)
        plt.imshow(tir_SR,cmap="gray")
        plt.title("TIR SR")
        plt.axis("off")
        i += 1

    if real_rgb is not None:
        plt.subplot(1, cols, i)
        plt.imshow(real_rgb)
        plt.title("Generated RGB")
        plt.axis("off")
        

    plt.tight_layout()
    plt.show()

    

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    modelSR = load_modelSR()
    modelCol = load_modelCol()

    print("✅ model loaded")


    tir_path = r"output/patches/lucknow/sample_007/tir_200m.npy"


    physics = load_sample(tir_path)
    tir_SR = predictSR(modelSR, physics)


    print("✅  TIR super resolution done")


    tir_SR=tir_SR*(TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6) +TIR_GLOBAL_MIN 
    



    physics, _ = load_sample2(tir_SR)
    RGB_Formate = predictCol(modelCol, physics)
    print("✅ RGB generated")



    tir_SR= (tir_SR- TIR_GLOBAL_MIN) / (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
    tir_SR= np.clip(tir_SR, 0.0, 1.0)


    RGB_Formate= np.clip(RGB_Formate, 0.0, 1.0)
    RGB_Formate= (RGB_Formate-RGB_Formate.min()) / (RGB_Formate.max()-RGB_Formate.min() + 1e-6 ) 



    # tir_img = physics.squeeze().numpy()[0]

    tir_img = np.load(tir_path).astype(np.float32)
    tir_img = np.squeeze(tir_img)

    tir_img = (tir_img - TIR_GLOBAL_MIN )/ (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
    tir_img= np.clip(tir_img, 0.0, 1.0)




    tir_img=percentile_stretch(tir_img)
    tir_SR=percentile_stretch(tir_SR)
    RGB_Formate=percentile_stretch(RGB_Formate)
    


    
    tir_img  =  tir_img/255.0
    tir_SR  =  tir_SR*0.9/255.0
    RGB_Formate  =  RGB_Formate /255.0

    RGB_Formate  =  RGB_Formate *0.5 


    visualize_numpy(tir_img, tir_SR,RGB_Formate)
    print("✅ Done - PNG + visualization saved")




    img = cv2.imread('results/SR.png')
   
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Apply Gaussian Blur 
    gaussian_blur = cv2.GaussianBlur(img_rgb, (3,3), 0)

    # 3. Apply Median Blur
    median_blur = cv2.medianBlur(img_rgb, 3)

    # 4. Apply Bilateral Filtering
    bilateral_filter = cv2.bilateralFilter(img_rgb, 3, 75, 75)

    # 5. Plot and compare results
    images = [img_rgb, gaussian_blur, median_blur, bilateral_filter]
    titles = ['Original (Noisy)', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']



    plt.figure(figsize=(12, 12))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    
    # 1. Load the image (replace 'image_f94f83.jpg' with your actual file path)
    img = cv2.imread('results/RGB.png')
    # OpenCV reads images as BGR, convert it to RGB for proper display in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Apply Gaussian Blur 
    # (3,3) is the kernel size (must be odd numbers). Larger numbers = more blur.
    gaussian_blur = cv2.GaussianBlur(img_rgb, (5,5), 1.5)

    # 3. Apply Median Blur
    # 5 is the kernel size. Great for removing the grid pattern.
    median_blur = cv2.medianBlur(img_rgb, 3)

    # 4. Apply Bilateral Filtering
    # 9 = Pixel neighborhood diameter, 75 = color sigma, 75 = spatial sigma
    bilateral_filter = cv2.bilateralFilter(img_rgb, 3, 75, 75)

    # 5. Plot and compare results
    images = [img_rgb, gaussian_blur, median_blur, bilateral_filter]
    titles = ['Original (Noisy)', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter']

    plt.figure(figsize=(12, 12))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



    print("✅ Done ")