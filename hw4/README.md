# NYCU-VRDL-2025-Spring-HW4
StudentID: 313553047  
Name: 吳佩怡

## Introduction
Image restoration base on PromptIR model, modifications made to be suitable for the given dataset

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
Train the model, save model checkpoints and evaluate loss and psnr for each epoch
```
python train.py
```
Test the model with provided dataset, results saved as zip file containing pred.npz  
Visualized results can be saved as PNGs in desired location

```
python test.py
```

## Performance Snapshot
