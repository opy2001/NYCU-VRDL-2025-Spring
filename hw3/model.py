
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def get_model(num_classes, ckpt=None, resume=False, layers=None):
    # layers: default=3 (if set None)

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=layers
    )
    
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes) 
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    if resume == True:
        model.load_state_dict(torch.load(ckpt))
        print(f'>>>>> Loaded model {ckpt}')

    return model 


def get_lr(model, optim_type='SGD', num_epochs=50):

    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=0.0025, 
            momentum=0.9, 
            weight_decay=1e-4
        )
        scheduler = MultiStepLR(optimizer, milestones=[int(0.6*num_epochs), int(0.8*num_epochs)], gamma=0.1)

    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    

    return optimizer, scheduler



    
