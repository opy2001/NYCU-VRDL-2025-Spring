import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_class = 100
batch_size = 32
num_workers = 8


transform = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.3),  
    nn.Linear(model.fc.in_features, n_class)
)
checkpoint = torch.load("model.pth.tar")  # model path
model.load_state_dict(checkpoint)
model.to(device)

img_folder = "./data/test"
img_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', 'png'))]

class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

model.eval()
results = []

with torch.no_grad():
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0).to(device)
        res = model(img)
        _, pred_class = res.max(1)

        results.append([img_name[:-4], class_names[pred_class.item()]])


df = pd.DataFrame(results, columns=["image_name", "pred_label"])
df.to_csv("prediction.csv", index=False)

