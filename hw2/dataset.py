import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomPerspective

class DATASET(Dataset):
    def __init__(self, img_dir, json_path):
        self.img_dir = img_dir

        if img_dir[-5:] == 'train':
            self.transforms = Compose([
                RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
                ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = data['images']
        self.anno_list = data['annotations']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        image_id = self.images[i]['id']
        image_path = f"{self.img_dir}/{self.images[i]['file_name']}"

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        annos = [anno for anno in self.anno_list if anno['image_id'] == image_id]
        boxes = [anno['bbox'] for anno in annos]
        labels = [anno['category_id'] for anno in annos]

        target = {
            'boxes': torch.tensor([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes]),  # xmin, ymin, xmax, ymax
            'labels': torch.tensor(labels)
        }

        return image, target


class TESTSET(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        image_id = self.images[i][:-4]
        image_path = f'{self.img_dir}/{image_id}.png'

        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        return image, int(image_id)