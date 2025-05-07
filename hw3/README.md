# NYCU-VRDL-2025-Spring-HW3
StudentID: 313553047  
Name: 吳佩怡

## Introduction
Instance segmentation with Mask RCNN, implemented with ResNet-50 backbone

## How to Install
Clone the repository and install dependencies
```
git clone https://github.com/opy2001/NYCU-VRDL-2025-Spring.git
pip install requirements.txt
```
(Optional) Create and run under virtual environment
```
conda create --name env_name python=3.9
conda activate env_name
```
Train the model, save model checkpoints and evaluate mAP for each epoch
```
python train.py
```
Test the model with provided dataset, results saved as zip file containing test-results.json 
```
python test.py
```

## Performance Snapshot
<img width="1157" alt="Screenshot 2025-05-06 at 2 22 15 PM" src="https://github.com/user-attachments/assets/a0397e23-5ac3-4bc1-8bbf-3e9f63e33cec" />
