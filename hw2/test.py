import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json, csv
import zipfile

from dataset import TESTSET
from model import get_model

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def test(model, test_loader, json_path, csv_path, score_low, score_mid, score_high):
    model.eval()
    task1 = []
    task2 = []

    for images, image_ids in tqdm(test_loader, desc='====== Producing test results ====='):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for image_id, output in zip(image_ids, outputs):
            image_id = int(image_id) # tensor -> int

            # ==== tolist() only works on cpu ====
            boxes = output['boxes'].cpu().tolist()
            scores = output['scores'].cpu().tolist()
            cat_ids = output['labels'].cpu().tolist()

            # ==== task2, filtered bboxes ====
            true_boxes = []
            true_high= []
            true_mid = []
            true_low = []
            # ===== task1, append bbox info to json =====
            for bbox, score, cat_id in zip(boxes, scores, cat_ids):
                xmin, ymin, xmax, ymax = bbox
                new_box = [xmin, ymin, xmax-xmin, ymax-ymin] # xmin, ymin, w, h
                pred = {
                    'image_id': image_id,
                    'bbox': new_box,
                    'score': score,
                    'category_id': cat_id
                }
                task1.append(pred)
                # ==== task2, filter by score ====
                if score >= score_low:
                    true_low.append(pred)
                    if score >= score_mid:
                        true_mid.append(pred)
                        if score >= score_high:
                            true_high.append(pred)
                if len(true_high) != 0: true_boxes = true_high
                elif len(true_mid) != 0: true_boxes = true_mid
                elif len(true_low) != 0: true_boxes = true_low

            # ==== sort by bbox xmin for each image ====================
            true_boxes = sorted(true_boxes, key=lambda x:x['bbox'][0])
            pred_label = -1
            if len(true_boxes) > 0:
                pred_label = int(''.join(str(int(x['category_id'])-1) for x in true_boxes))
            task2.append([image_id, pred_label])

    # ==== task 1 ====
    with open(json_path, 'w') as f:
        json.dump(task1, f, indent=4)
    # ==== task 2 ====  
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id','pred_label'])
        for t in task2:
            writer.writerow(t)

    
def main():

    num_classes = 11  # +background

    # ===== test data =====
    test_img_dir = 'nycu-hw2-data/test'
    test_data = TESTSET(test_img_dir)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # ===== model =====
    ckpt = 'dynam4/e13.pth'
    model = get_model(num_classes, ckpt=ckpt, model_type=2) # 2->res50, 3->mobilenet
    model.to(device)

    # ===== test =====
    json_path = 'pred.json'
    csv_path = 'pred.csv'
    # == task2 score threshold ==
    score_low = 0.3
    score_mid = 0.6
    score_high = 0.8
    test(model, test_loader, json_path, csv_path, score_low, score_mid, score_high)

    # ===== zip json & csv =====
    zip_fname = 'dyn13.zip'
    with zipfile.ZipFile(zip_fname, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path)
        zf.write(csv_path)
   

if __name__ == "__main__":
    main()

