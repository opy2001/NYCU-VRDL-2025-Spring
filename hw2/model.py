import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def get_model(num_classes, ckpt=None, model_type=0, layers=None):
    # ==================================
    # 0: startup res50
    # 1: startup mobilenet
    # 2: ckpt res50
    # 3: ckpt mobilenet
    # layers: default=3 (if set None)
    # ==================================
    if model_type%2 == 0: 
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=layers
        )
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=layers
        )
    
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes) 
    
    if model_type > 1:
        model.load_state_dict(torch.load(ckpt))
        print(f'>>>>> Loaded model {ckpt}')

    return model 

def get_optim(model, optim_type='SGD'):
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=5e-3, 
            momentum=0.9, 
            weight_decay=1e-4
        )
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
    elif optim_type == 'Dynamic':
        backbone_lr = 1e-4
        head_lr = 3e-4
        params = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.rpn.parameters(), "lr": head_lr},
            {"params": model.roi_heads.parameters(), "lr": head_lr},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

    return optimizer

def get_scheduler(optimizer, num_epochs=20, dynam=False):
    # ==== optim_type 'Dynamic' ==== 
    if dynam:
        warmup_epochs = 3

        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)

        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup, cosine], 
            milestones=[warmup_epochs]
        )
        return scheduler
    # ==== optim_type 'SGD' or 'AdamW' ====
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


    

