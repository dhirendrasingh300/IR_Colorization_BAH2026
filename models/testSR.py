import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from generatorSR import Generator




# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/gen_SR_100.pth"
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

def load_model():
    model = Generator(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


# =========================================================
# LOAD SAMPLE
# =========================================================

def load_sample(tir_path, rgb_path=None):

    

    tir = np.load(tir_path).astype(np.float32)
    tir_hr = np.load(rgb_path).astype(np.float32)

    

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
    tir_hr = np.expand_dims(tir_hr, axis=0) 
    print(physics.shape,tir_hr.shape)

    return physics, tir_hr


# =========================================================
# INFERENCE
# =========================================================

def predict(model, physics):
    with torch.no_grad():

        # convert numpy → tensor
        physics = torch.tensor(physics, dtype=torch.float32)

        # 🔥 FORCE 4D SHAPE
        if physics.dim() == 2:
            physics = physics.unsqueeze(0).unsqueeze(0)   # H,W → 1,1,H,W

        elif physics.dim() == 3:
            physics = physics.unsqueeze(0)  # C,H,W → 1,C,H,W

        physics = physics.to(DEVICE)

        out = model(physics)

    return out.cpu().squeeze(0).numpy()

def percentile_stretch(image, low=2, high=98):
    """
    Stretches the intensity of an image based on percentiles to remove outliers.
    """
    
    if image.ndim == 3:
        stretched = np.zeros_like(image)
        for i in range(image.shape[-1]):
            stretched[..., i] = percentile_stretch(image[..., i], low, high)
        return stretched.astype(np.uint8)
    
    low_val = np.percentile(image, low)   
    # while low_val==0:
    #     low=low +2
    #     low_val = np.percentile(image, low) 
        
    high_val = np.percentile(image, high)
    
    stretched = np.clip(image, low_val, high_val)
    stretched = (stretched - low_val) * 255.0 / (high_val - low_val + 1e-5)
    
    return stretched.astype(np.uint8)
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

def save_png(fake_rgb, save_path,name):
    """
    Save normalized RGB image as PNG.

    Parameters
    ----------
    fake_rgb : numpy.ndarray
        Shape: (3,H,W) or (H,W,3)
        Range: [0,1]
    save_path : str
        Example: "results/fake.png"
    """
    save_path=save_path +"/"+ name+".png"

    img = np.array(fake_rgb)

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

    print(f"✅ PNG saved: {save_path}")


# =========================================================
# VISUALIZATION
# =========================================================




import numpy as np
import matplotlib.pyplot as plt

def visualize_numpy(tir, real_rgb=None, fake_rgb=None):

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
    fake_rgb = prepare(fake_rgb)


    save_png(fake_rgb, save_path,"fake")
    save_png(real_rgb, save_path,"real")
    save_png(tir, save_path,"tir")

    print(fake_rgb.shape,type(fake_rgb))
    print(tir_img.shape,type(tir_img))
    real_rgb=clean(real_rgb)
    print(real_rgb.shape,type(real_rgb))

    # print("Fake RGB :", fake_rgb.shape, type(fake_rgb),
    #   "Min =", np.min(fake_rgb), "Max =", np.max(fake_rgb))

    # print("TIR Image:", tir_img.shape, type(tir_img),
    #     "Min =", np.min(tir_img), "Max =", np.max(tir_img))

    # real_rgb = clean(real_rgb)

    # print("Real RGB :", real_rgb.shape, type(real_rgb),
    #     "Min =", np.min(real_rgb), "Max =", np.max(real_rgb))

    cols = 1 + (real_rgb is not None) + (fake_rgb is not None)

    plt.figure(figsize=(5 * cols, 5))

    i = 1
    plt.subplot(1, cols, i)
    plt.imshow(tir, cmap='gray')
    plt.title("Input TIR")
    plt.axis("off")
    i += 1

    if real_rgb is not None:
        plt.subplot(1, cols, i)
        plt.imshow(real_rgb,cmap="gray")
        plt.title("Ground Truth RGB")
        plt.axis("off")
        i += 1

    if fake_rgb is not None:
        plt.subplot(1, cols, i)
        plt.imshow(fake_rgb,cmap="gray")
        plt.title("Generated RGB")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    model = load_model()

    tir_path = r"output/patches/jammu/sample_014/tir_200m.npy"
    rgb_path = r"output/patches/jammu/sample_014/rgb_100m_512.npy"

    # tir_path = r"output/patches/demo/sample_006/tir_100m_512.npy"
    # rgb_path = r"output/patches/demo/sample_006/rgb_100m_512.npy"






    physics, real_rgb = load_sample(tir_path, rgb_path)

    print(physics.shape,real_rgb.shape)

    fake_rgb = predict(model, physics)

    rgb = np.array(fake_rgb)

    print(rgb.shape)
    print(rgb.dtype)
    print("Min:", rgb.min())
    print("Min:", rgb.mean())
    print("Max:", rgb.max())
    print("Unique:", np.unique(rgb)[:20])



   
    fake_rgb= np.clip(fake_rgb, 0.0, 1.0)
    fake_rgb= (fake_rgb-fake_rgb.min()) / (fake_rgb.max()-fake_rgb.min() + 1e-6 ) 


    rgb = np.array(fake_rgb)

    print(rgb.shape)
    print(rgb.dtype)
    print("Min:", rgb.min())
    print("Max:", rgb.max())
    print("Unique:", np.unique(rgb)[:20])


    
    rgb = np.array(real_rgb)

    print(rgb.shape)
    print(rgb.dtype)
    print("Min:", rgb.min())
    print("Max:", rgb.max())
    print("Unique:", np.unique(rgb)[:20])



    # tir_img = physics.squeeze().numpy()[0]

    tir_img = np.load(tir_path).astype(np.float32)
    tir_img = np.squeeze(tir_img)

    tir_img = (tir_img - TIR_GLOBAL_MIN )/ (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN + 1e-6)
    tir_img= np.clip(tir_img, 0.0, 1.0)




    print("TIR Image:", tir_img.shape, type(tir_img),
        "Min =", np.min(tir_img), "Max =", np.max(tir_img))


    tir_img=percentile_stretch(tir_img)
    

    real_rgb=percentile_stretch(np.array(real_rgb))
    fake_rgb=percentile_stretch(fake_rgb)

    
    tir_img=tir_img/255.0
    real_rgb=real_rgb/255.0
    fake_rgb=fake_rgb/255.0
    target_mean= real_rgb.mean()

    print("means =",target_mean,fake_rgb.mean())
    
    # fake_rgb=fake_rgb-fake_rgb.mean()+ target_mean
    print("means =",fake_rgb.mean())
     

    


    
    visualize_numpy(tir_img, real_rgb, fake_rgb)

    import cv2
    import matplotlib.pyplot as plt

    # 1. Load the image (replace 'image_f94f83.jpg' with your actual file path)
    img = cv2.imread('results/fake.png')
    # OpenCV reads images as BGR, convert it to RGB for proper display in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Apply Gaussian Blur 
    # (3,3) is the kernel size (must be odd numbers). Larger numbers = more blur.
    gaussian_blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)

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



    # show_results(tir_img, fake_rgb, real_rgb)

    # save_png(fake_rgb)

    print("✅ Done - PNG + visualization saved")