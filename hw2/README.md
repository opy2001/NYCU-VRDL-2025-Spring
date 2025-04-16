
# NYCU-VRDL-2025-Spring-HW2
StudentID: 313553047  
Name: 吳佩怡

## Introduction
Digit recognition with Faster RCNN, implemented with ResNet-50 backbone with FPN

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
Test the model with provided image dataset, results saved as zip file containing pred.json and pred.csv
```
python test.py
```

## Performance Snapshot
![snapshot](https://github.com/user-attachments/assets/b3fcbfbf-63ce-4aee-a60e-83a5df1ed309)

