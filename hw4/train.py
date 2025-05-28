import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio

from dataset import DATASET
from model import *
from utils import plot_curve

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# =================================================================
def train(model, train_loader, optimizer, scheduler, scaler, criterion):
    model.train()
    train_loss = 0.0

    for image, target in train_loader:
        #print(image)
        #print(target)
        image = image.to(device)
        target = target.to(device)
        #images = [img.to(device) for img in images]
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        optimizer.zero_grad()

        with autocast('cuda'):  
            output = model(image)
            loss = criterion(output, target)

        scaler.scale(loss).backward()      
        scaler.step(optimizer)             
        scaler.update()                    
        train_loss += loss.item()
    
    #scheduler.step() 

    return train_loss / len(train_loader)
# ===============================================================
@torch.no_grad()
def validate(model, val_loader, optimizer, criterion):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    for image, target in val_loader:
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(image)
        # ==== loss ====
        loss = criterion(output, target)
        val_loss += loss.item()
        # ==== psnr ====
        output = np.clip(output.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
        target = np.clip(target.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
        psnr = 0
        for i in range(output.shape[0]):
            psnr += peak_signal_noise_ratio(target[i], output[i], data_range=1)
        val_psnr += psnr/output.shape[0]

    return val_loss/len(val_loader), val_psnr/len(val_loader)
    

# =======================================================
def main():
    # ===== parameters =====
    num_epochs = 100
    batch_size = 1
    num_workers = 4
    save_dir = 'mix_loss'

    # ===== data directories =====
    root_dir = 'hw4_dataset/train'
    all_idx = [x for x in range(1, 1601)]
    train_idx, val_idx = train_test_split(all_idx, test_size=0.1, random_state=42)
    train_imgs = [f'rain-{i}' for i in train_idx] + [f'snow-{i}' for i in train_idx]
    val_imgs = [f'rain-{i}' for i in val_idx] + [f'snow-{i}' for i in val_idx]
    # ===== load data =====
    train_data = DATASET(root_dir, train_imgs, train=True)
    val_data = DATASET(root_dir, val_imgs, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=num_workers, shuffle=False)
    # ===== model =====================
    start_epoch = 0
    #ckpt = 'aug/e80.pth'
    model = PromptIR(decoder=True)
    #model.load_state_dict(torch.load(ckpt)) 
    model.to(device)

    # ===== loss function options ======
    # nn.L1Loss()
    # CharbonnierLoss()
    # SSIMLoss()
    # CombinedLoss()
    # ==================================
    criterion = CombinedLoss().to(device)

    # ===== LR ==========================
    optimizer, scheduler = get_lr(model)  

    # ===== train & val ===================================================
    loss_curve = []
    vloss_curve = []
    psnr_curve = []

    scaler = GradScaler()
    for epoch in range(start_epoch, num_epochs):
        
        print(f'=============== Start training epoch {epoch+1} ===============')
        train_loss = train(model, train_loader, optimizer, scheduler, scaler, criterion)
        loss_txt = f'{save_dir}/loss.txt'
        with open(loss_txt, 'a') as f:
            f.write(f'Loss @ Epoch {epoch+1}: {train_loss}\n')

        vloss, psnr = validate(model, val_loader, optimizer, criterion)
        vloss_txt = f'{save_dir}/vloss.txt'
        psnr_txt = f'{save_dir}/psnr.txt'
        with open(vloss_txt, 'a') as f:
            f.write(f'Validation Loss @ Epoch {epoch+1}: {vloss}\n')
        with open(psnr_txt, 'a') as f:
            f.write(f'PSNR @ Epoch {epoch+1}: {psnr}\n')
        
        scheduler.step(vloss)

        # ===== plot curves =====
        loss_curve.append(train_loss)
        vloss_curve.append(vloss)
        psnr_curve.append(psnr)

        print(f"=========== Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {vloss} | PSNR: {psnr} ===========")

        torch.save(model.state_dict(), f'{save_dir}/e{epoch+1}.pth')


    # ===== plot curves =====
    plot_curve(save_dir, loss_curve, vloss_curve, psnr_curve)

if __name__ == "__main__":
    main()