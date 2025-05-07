import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DATASET(Dataset):
    def __init__(self, root_dir, folders, train=True):
        self.root_dir = root_dir 
        self.folders = folders
        self.train = train

    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        folder_name = self.folders[idx]
        folder_path = f"{self.root_dir}/{folder_name}"

        image_path = f"{folder_path}/image.tif"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_names = sorted([x for x in os.listdir(folder_path) if x[0]=='c'])
        masks = []
        labels = []
        for mask_name in mask_names:
            class_id = int(mask_name[5])
            mask_image = cv2.imread(f'{folder_path}/{mask_name}', cv2.IMREAD_UNCHANGED)
            instances = np.unique(mask_image)
            instances = [x for x in instances if x!= 0]
            for ins in instances:
                binary_mask = (mask_image == ins).astype(np.uint8)
                masks.append(binary_mask)
                labels.append(class_id)

        H, W = image.shape[:2]
        if not masks:
            masks = np.zeros((0, H, W), dtype=np.uint8)
        else:
            masks = np.stack(masks)  # [N, H, W]

        transform = self._add_transform(len(masks))

        # ==== dict format for albumenation ====
        data = {"image": image}
        for i, m in enumerate(masks):
            data[f"mask{i}"] = m

        # ==== transform ====
        transformed = transform(**data)
        image = transformed["image"]

        transformed_masks = []
        for i in range(len(masks)):
            m = transformed.get(f"mask{i}")
            if m is not None:
                if isinstance(m, np.ndarray):
                    transformed_masks.append(torch.from_numpy(m))
                else:
                    transformed_masks.append(m)

        if transformed_masks:
            masks = torch.stack(transformed_masks)
        else:
            masks = torch.zeros((0, H, W), dtype=torch.uint8)

        # ==== masks to boxes ==== 
        valid_masks, new_labels = [], []
        for i, m in enumerate(masks):
            if torch.nonzero(m).numel() == 0:
                continue
            box = masks_to_boxes(m.unsqueeze(0))[0]         
            x1, y1, x2, y2 = box
            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue  
            valid_masks.append(m)
            new_labels.append(labels[i])

        if valid_masks:
            valid_masks = torch.stack(valid_masks)
            boxes = masks_to_boxes(valid_masks)
        else:
            valid_masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)


        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(new_labels, dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        return image, target
    
    def _add_transform(self, num_masks):
        additional_targets = {f"mask{i}": "mask" for i in range(num_masks)}

        if self.train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
                A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ], additional_targets=additional_targets)
        else:
            return A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2()
            ], additional_targets=additional_targets)
        
