import os
import numpy as np
import torch
from tqdm import tqdm
import zipfile
from torch.utils.data import DataLoader

from model import *
from dataset import TESTSET
from utils import npz_to_png

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test(model, test_loader, npz_save):
    model.eval()
    results = {}
    for img, img_name in tqdm(test_loader, desc="====== Producing test results ======"):
        img = img.to(device)
        output = model(img)
        
        for i in range(output.shape[0]):
            img_array = output[i].detach().cpu().numpy()     
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8) 
            results[img_name[i]] = img_array

    np.savez(npz_save, **results)


def main():

    # ===== test data =====
    root_dir = 'hw4_dataset/test/degraded'

    # ===== model =====
    ckpt = 'ssim/e50.pth'
    model = PromptIR(decoder=True)
    model.load_state_dict(torch.load(ckpt))
    print(f'>>>>> Loaded model {ckpt}')
    model.to(device)

    # ===== test result =====
    npz_save = 'pred.npz'
    test_data = TESTSET(root_dir)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    test(model, test_loader, npz_save)

    # ===== zip npz =====
    zip_fname = 's50.zip'
    with zipfile.ZipFile(zip_fname, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_save)

    # ===== save predicted png =====
    pics_save = ckpt.split('/')[0]
    npz_to_png(npz_save, f'{pics_save}/pics')

if __name__ == "__main__":
    main()
