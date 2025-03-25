import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import random
from visualize import Plot, Conf

# maintain same test setting
#seed = 42
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_class = 100
batch_size = 32
num_workers = 8
num_epochs = 35

# data augmentation 
transform_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(), 
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = ImageFolder(root="./data/train", transform=transform_aug)
val_data = ImageFolder(root="./data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, n_class)
)
model.to(device)

# unfreeze layer 3, 4, FC
for param in model.parameters():
    param.requires_grad = False 
for param in model.layer3.parameters():  
    param.requires_grad = True
for param in model.layer4.parameters():  
    param.requires_grad = True
for param in model.fc.parameters():  
    param.requires_grad = True

# update BN
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.train()  

criterion_train = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_val = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW([
    {"params": model.layer3.parameters(), "lr": 1e-4}, 
    {"params": model.layer4.parameters(), "lr":5e-4},  
    {"params": model.fc.parameters(), "lr": 1e-3} 
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)


Tloss = []
Tacc = []
Vloss = []
Vacc = []
val_labels = []
val_preds = []
lr_fc = []
lr_l4 = []
lr_l3 = []

best_val = 0

for epoch in range(num_epochs):
    
    model.train()
    train_loss = 0
    train_correct, train_total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        optimizer.zero_grad()
        loss = criterion_train(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predict = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predict.eq(labels).sum().item()

    model.eval()
    val_loss = 0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion_val(outputs, labels)

            val_loss += loss.item()
            _, predict = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predict.eq(labels).sum().item()
            val_preds.extend(predict.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    print("Epoch:{} Train_loss:{:.4f} Train_acc:{:.4f} Val_loss:{:.4f} Val_acc:{:.4f}".format(epoch+1,
                                                                                              train_loss/len(train_loader),
                                                                                              train_correct/train_total,
                                                                                              val_loss/len(val_loader),
                                                                                              val_correct/val_total))
    
    Tloss.append(train_loss/len(train_loader))
    Tacc.append(train_correct/train_total)
    Vloss.append(val_loss/len(val_loader))
    Vacc.append(val_correct/val_total)
    lr_l3.append(optimizer.param_groups[0]['lr'])
    lr_l4.append(optimizer.param_groups[1]['lr'])
    lr_fc.append(optimizer.param_groups[2]['lr'])

    if best_val < val_correct/val_total:
        torch.save(model.state_dict(), "model_best.pth.tar")
        best_val = val_correct/val_total

    if epoch%5 == 4:
       torch.save(model.state_dict(), "model_{}.pth.tar".format(epoch+1))
    
    scheduler.step()

Plot(Tloss, Vloss, Tacc, Vacc, num_epochs, lr_l3, lr_l4, lr_fc)
Conf(val_labels, val_preds)
