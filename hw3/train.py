
import os
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tempfile
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import DATASET
from model import *
from utils import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# =================================================================
def train(model, train_loader, optimizer, scheduler, scaler):
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast('cuda'):  
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        scaler.scale(loss).backward()      
        scaler.step(optimizer)             
        scaler.update()                    
        train_loss += loss.item()
    
    scheduler.step() 

    return train_loss / len(train_loader)
# ===============================================================
@torch.no_grad()
def compute_mAP(model, val_loader):
    model.eval()
    result = []
    gt = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': f'class{i}'} for i in range(1,5)] # class 1-4
    }

    anno_id = 1
    for images, targets in tqdm(val_loader, desc='Validating: '):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = model(images)
        
        for idx, pred in enumerate(predictions):
            target = targets[idx]
            image_id = int(target['image_id'].item())
            # ==== GT images ====
            # (N, C, H, W)
            w = images[idx].shape[-1]
            h = images[idx].shape[-2]
            gt['images'].append({
                'id': image_id,
                'file_name': f'{image_id}.jpg',
                'height': h,
                'width': w
            })
            # ==== GT masks ====
            gt_masks = target['masks'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            for i in range(len(gt_masks)):
                mask = gt_masks[i]
                encoded_mask = encode_mask(mask)
                gt['annotations'].append({
                    'id': anno_id,
                    'image_id': image_id,
                    'category_id': int(gt_labels[i]),
                    'segmentation': encoded_mask,
                    'area': int(mask.sum()),
                    'bbox': list(mask_utils.toBbox(encoded_mask)),
                    'iscrowd': 0
                })
                anno_id += 1
            # ==== predictions ====
            pred_masks = pred['masks'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            for i in range(len(pred_masks)):
                if pred_scores[i] < 0.5:
                    continue
                mask = pred_masks[i, 0] > 0.5
                encoded_mask = encode_mask(mask)
                result.append({
                    'image_id': image_id,
                    'category_id': int(pred_labels[i]),
                    'segmentation': encoded_mask,
                    'score': float(pred_scores[i])
                })
    # ==== COCO evals ======================================
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as gt_file, tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as pred_file:
        json.dump(gt, gt_file)
        json.dump(result, pred_file)
        gt_file.flush()
        pred_file.flush()

        cocoGt = COCO(gt_file.name)
        cocoDt = cocoGt.loadRes(pred_file.name)
        CE = COCOeval(cocoGt, cocoDt, iouType='segm')
        CE.evaluate()
        CE.accumulate()
        CE.summarize()

    return CE.stats[0]   # mAP @ [0.5 : 0.95]

# =======================================================
def main():
    # ===== parameters =====
    num_epochs = 70
    batch_size = 2
    num_classes = 5  # +background
    num_workers = 4
    save_dir = 'tr5_sgd_aug/'

    # ===== data directories =====
    root_dir = 'train_release'
    folders = os.listdir(root_dir)
    train_folders, val_folders = train_test_split(folders, test_size=0.1, random_state=42)
    # ===== load data =====
    train_data = DATASET(root_dir, train_folders, train=True)
    val_data = DATASET(root_dir, val_folders, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_data, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    # ===== model =====================
    start_epoch = 0
    ckpt = None
    model = get_model(num_classes, ckpt=ckpt, layers=5) 
    model.to(device)

    # ===== LR ==============================
    optimizer, scheduler = get_lr(model, num_epochs)  

    # ===== train & val ===================================================
    loss_curve = []
    mAP_curve = []

    scaler = GradScaler()
    for epoch in range(start_epoch, num_epochs):
        
        print(f'=============== Start training epoch {epoch+1} ===============')
        train_loss = train(model, train_loader, optimizer, scheduler, scaler)
        loss_txt = f'{save_dir}loss.txt'
        with open(loss_txt, 'a') as f:
            f.write(f'Loss @ Epoch {epoch+1}: {train_loss}\n')

        mAP = compute_mAP(model, val_loader)
        mAP_txt = f'{save_dir}mAP.txt'
        with open(mAP_txt, 'a') as f:
            f.write(f'mAP @ Epoch {epoch+1}: {mAP}\n')

        # ===== plot curves =====
        loss_curve.append(train_loss)
        mAP_curve.append(mAP)


        print(f"=========== Epoch {epoch+1} | Train Loss: {train_loss:.4f} | mAP: {mAP} ===========")

        torch.save(model.state_dict(), f'{save_dir}e{epoch+1}.pth')


    # ===== plot curves =====
    plot_curve(save_dir, loss_curve, mAP_curve)

if __name__ == "__main__":
    main()
