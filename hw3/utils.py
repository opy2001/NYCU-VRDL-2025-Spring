import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array


def plot_curve(save_dir, loss, mAP):
    # ===== train loss =====
    plt.figure(figsize=(6,5))
    plt.plot(loss, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train Loss")
    plt.savefig(f"{save_dir}loss.png")
    # ===== valid mAP =====
    plt.figure(figsize=(6,5))
    plt.plot(mAP, label='Validation mAP')
    plt.xlabel("Epoch")
    plt.ylabel("AP @ IoU [0.5:0.95]")
    plt.legend()
    plt.title("Validation mAP")
    plt.savefig(f"{save_dir}mAP.png")
