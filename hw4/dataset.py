from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
from torchvision.transforms import functional as F

class DATASET(Dataset):
    def __init__(self, root_dir, img_names, train=True):
        self.root_dir = root_dir
        self.img_names = img_names
        self.train = train

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = f'{self.img_names[idx]}.png'
        name_list = img_name.split("-")
        clean_path = f'{self.root_dir}/clean/{name_list[0]}_clean-{name_list[1]}'
        degraded_path = f'{self.root_dir}/degraded/{img_name}'

        clean_img = Image.open(clean_path).convert("RGB")
        degraded_img = Image.open(degraded_path).convert("RGB")

        # ==== augmentation for training set ====
        if self.train:
            if random.random() > 0.5:
                clean_img = F.hflip(clean_img)
                degraded_img = F.hflip(degraded_img)

            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                clean_img = F.rotate(clean_img, angle)
                degraded_img = F.rotate(degraded_img, angle)
        
        # ===================================
        transform = transforms.ToTensor()
        clean_img = transform(clean_img)
        degraded_img = transform(degraded_img)

        #if self.train:
        degraded_img = self._random_crop(degraded_img, crop_size=256)
        clean_img = self._random_crop(clean_img, crop_size=256)

        return degraded_img, clean_img
    
    def _random_crop(self, img, crop_size):
        _, h, w = img.shape  # (C, H, W)

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        img_crop = img[:, top:top + crop_size, left:left + crop_size]

        return img_crop


# ===================================================
class TESTSET(Dataset):
    def __init__(self, root_dir):
        self.img_dir = root_dir
        self.img_names = [x for x in os.listdir(root_dir)]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        transform = transforms.ToTensor()
        img = transform(img)

        return img, img_name