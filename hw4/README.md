# NYCU-VRDL-2025-Spring-HW4
StudentID: 313553047  
Name: 吳佩怡

## Introduction
Image restoration based on PromptIR model, modifications made to be suitable for the given dataset

## How to Install
Clone the repository and install dependencies
```
git clone https://github.com/opy2001/NYCU-VRDL-2025-Spring.git
pip install -r requirements.txt
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
<img width="1080" alt="Screenshot 2025-05-28 at 2 48 32 PM" src="https://github.com/user-attachments/assets/c5a9dfda-4dc1-4c54-abff-2a00dbbff93d" />
