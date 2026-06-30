
from utils.visualization import percentile_stretch
from Run_model import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATASET = "output/patches/jammu"



modelSR = load_modelSR()
modelCol = load_modelCol()

mse_list = []
mae_list = []
psnr_list = []
ssim_list = []

for sample in sorted(os.listdir(DATASET)):

    folder = os.path.join(DATASET, sample)

    tir_path = os.path.join(folder, "tir_200m.npy")
    rgb_path = os.path.join(folder, "rgb_100m_512.npy")

    if not (os.path.exists(tir_path) and os.path.exists(rgb_path)):
        continue

    # -----------------------------
    # Prediction
    # -----------------------------

    tir = load_sample(tir_path)

    tir_SR = predictSR(modelSR, tir)

    tir_SR = tir_SR * (TIR_GLOBAL_MAX - TIR_GLOBAL_MIN) + TIR_GLOBAL_MIN


    physics2, gt_rgb = load_sample2(tir_SR, rgb_path)

    pred_rgb = predictCol(modelCol, physics2)

    # -----------------------------
    # Normalize
    # -----------------------------

    pred_rgb = np.clip(pred_rgb, 0, 1)

    gt_rgb = gt_rgb.squeeze(0).numpy()

    if gt_rgb.shape[0] == 3:
        gt_rgb = np.transpose(gt_rgb, (1,2,0))

    if pred_rgb.shape[0] == 3:
        pred_rgb = np.transpose(pred_rgb, (1,2,0))
    
    # pred_rgb=pred_rgb*(RGB_GLOBAL_MAX- RGB_GLOBAL_MIN + 1e-6) +RGB_GLOBAL_MIN 

    # -----------------------------
    # Metrics
    # -----------------------------

    mse = mean_squared_error(
        gt_rgb.flatten(),
        pred_rgb.flatten()
    )

    mae = mean_absolute_error(
        gt_rgb.flatten(),
        pred_rgb.flatten()
    )

    psnr = peak_signal_noise_ratio(
        gt_rgb,
        pred_rgb,
        data_range=1.0
    )

    ssim = structural_similarity(
        gt_rgb,
        pred_rgb,
        channel_axis=-1,
        data_range=1.0
    )

    mse_list.append(mse)
    mae_list.append(mae)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    print(f"{sample:20s}  PSNR={psnr:.2f}  SSIM={ssim:.4f}")

print("\n========================")
print("Evaluation Results")
print("========================")

print("Images :", len(psnr_list))
print("Average MSE :", np.mean(mse_list))
print("Average MAE :", np.mean(mae_list))
print("Average PSNR:", np.mean(psnr_list))
print("Average SSIM:", np.mean(ssim_list))

print(
    f"""
========================
Evaluation Results
========================
Images : {len(psnr_list)}

MSE  -> Avg: {np.mean(mse_list):.6f} | Min: {np.min(mse_list):.6f} | Max: {np.max(mse_list):.6f}
MAE  -> Avg: {np.mean(mae_list):.6f} | Min: {np.min(mae_list):.6f} | Max: {np.max(mae_list):.6f}
PSNR -> Avg: {np.mean(psnr_list):.4f} | Min: {np.min(psnr_list):.4f} | Max: {np.max(psnr_list):.4f}
SSIM -> Avg: {np.mean(ssim_list):.6f} | Min: {np.min(ssim_list):.6f} | Max: {np.max(ssim_list):.6f}
"""
)