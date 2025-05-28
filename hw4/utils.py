import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_curve(save_dir, loss, vloss, psnr):
    # ===== loss =====
    plt.figure(figsize=(6,5))
    plt.plot(loss, label='Train Loss')
    plt.plot(vloss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train & Val Loss")
    plt.savefig(f"{save_dir}/loss.png")
    # ===== val loss =====
    plt.figure(figsize=(6,5))
    plt.plot(psnr, label='PSNR')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.title("Validation PSNR")
    plt.savefig(f"{save_dir}/psnr.png")


def npz_to_png(npz_path, pic_save):
    os.makedirs(pic_save, exist_ok=True)
    data = np.load(npz_path)
    pname = pic_save.split('/')[0]

    for name in data.files:
        img_array = data[name]  # (3, H, W)
        img_array = np.transpose(img_array, (1, 2, 0))  # (H, W, 3)
        img = Image.fromarray(img_array)
        img.save(os.path.join(pic_save, f'{name[:-4]}_{pname}.png'))

    print(f'>>>>> Image saved to {pic_save}')
