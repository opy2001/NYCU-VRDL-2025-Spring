import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import DATASET
from model import *
from visualize import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    return train_loss / len(train_loader), optimizer.param_groups[0]['lr']
# ===============================================================
@torch.no_grad()
def compute_mAP(model, val_loader):
    model.eval()
    predictions = []
    gt = []

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions.extend(x for x in model(images))
        gt.extend(targets)

    metrics = MeanAveragePrecision(iou_type="bbox")
    metrics.update(preds=predictions, target=gt)
    map_dict = metrics.compute()
    mAP = map_dict['map']

    return mAP

# =======================================================
def main():
    # ===== parameters =====
    num_epochs = 20
    batch_size = 16 # per gpu if using DDP
    num_classes = 11  # +background
    num_workers = 4
    save_dir = 'final_results/'

    # ===== data directories =====
    train_img_dir = 'nycu-hw2-data/exp'
    train_json_dir = 'nycu-hw2-data/exp.json'
    val_img_dir = 'nycu-hw2-data/exp'
    val_json_dir = 'nycu-hw2-data/exp.json'
    # ===== load data ======
    train_data = DATASET(train_img_dir, train_json_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_data = DATASET(val_img_dir, val_json_dir)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    # ===== model =====================
    # 0: startup res50
    # 1: startup mobilenet
    # 2: ckpt res50
    # 3: ckpt mobilenet
    # layers: default=3 (if set None)
    # =================================
    start_epoch = 0
    ckpt = None
    model = get_model(num_classes, ckpt=ckpt, model_type=0, layers=4) 
    model.to(device)

    # ===== LR ==============================
    optimizer = get_optim(model, 'Dynamic')  
    scheduler = get_scheduler(optimizer, num_epochs, dynam=True)

    # ===== train & val ===================================================
    loss_curve = []
    lr_curve = []
    mAP_curve = []

    best_loss = 100.0000
    best_mAP = 0.0

    scaler = GradScaler()
    for epoch in range(start_epoch, num_epochs):
        
        print(f'=============== Start training epoch {epoch+1} ===============')
        train_loss, lr = train(model, train_loader, optimizer, scheduler, scaler)
        loss_txt = f'{save_dir}loss.txt'
        with open(loss_txt, 'a') as f:
            f.write(f'Loss @ Epoch {epoch+1}: {train_loss}\n')
        if train_loss < best_loss:
            best_loss = train_loss

        mAP = compute_mAP(model, val_loader)
        mAP_txt = f'{save_dir}mAP.txt'
        with open(mAP_txt, 'a') as f:
            f.write(f'mAP @ Epoch {epoch+1}: {mAP}\n')
        if mAP >  best_mAP:
            best_mAP = mAP

        # ===== plot curves =====
        loss_curve.append(train_loss)
        lr_curve.append(lr)
        mAP_curve.append(mAP)


        print(f"=========== Epoch {epoch+1} | Train Loss: {train_loss:.4f} | mAP: {mAP} ===========")

        torch.save(model.state_dict(), f'{save_dir}e{epoch+1}.pth')

        # ==== (optional) early stopping ====
        if epoch > 19 and mAP - best_mAP < 0.001 and best_loss - train_loss < 0.001:
            print(f'>>>>> Early stopping @ epoch {epoch+1}')
            break

    # ===== plot curves =====
    plot_curve(save_dir, loss_curve, lr_curve, mAP_curve)

if __name__ == "__main__":
    main()