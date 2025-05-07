
import os
import torch
from tqdm import tqdm
import json
import zipfile
from PIL import Image
import torchvision.transforms.functional as F

from model import get_model
from utils import encode_mask

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def test(model, root_dir, json_info, json_save, score_thr):
    model.eval()
    
    result = []
    for info in tqdm(json_info, desc="====== Producing test results ======"):
        file_name = info['file_name']
        image_id = info['id']
        image_path = f'{root_dir}/{file_name}'
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        outputs = model(image_tensor)[0]

        for i in range(len(outputs['scores'])):
            mask = outputs['masks'][i, 0].cpu().numpy() > score_thr
            encoded_mask = encode_mask(mask)

            result.append({
                'image_id': image_id,
                'bbox': outputs['boxes'][i].tolist(),
                'score': float(outputs['scores'][i]),
                'category_id': outputs['labels'][i].item(),
                'segmentation': encoded_mask
            })
        
        with open(json_save, 'w') as f:
            json.dump(result, f)

def main():

    num_classes = 5  # +background

    # ===== test data =====
    root_dir = 'test_release'
    json_dir = 'test_image_name_to_ids.json'
    with open(json_dir, 'r') as f:
        json_info = json.load(f)

    # ===== model =====
    ckpt = 'tr5_sgd_aug/e47.pth'
    model = get_model(num_classes, ckpt=ckpt, resume=True) 
    model.to(device)

    # ===== test result =====
    json_save = 'test-results.json'
    score_thr = 0.5
    test(model, root_dir, json_info, json_save, score_thr)

    # ===== zip json =====
    zip_fname = 'tr5_47.zip'
    with zipfile.ZipFile(zip_fname, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_save)


if __name__ == "__main__":
    main()
