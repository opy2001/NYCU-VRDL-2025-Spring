import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def Plot(Tloss, Vloss, Tacc, Vacc, num_epoch, lr_l3, lr_l4, lr_fc):
    # loss curve
    plt.figure(figsize=(5, 5))
    plt.plot(Tloss, label='Train Loss')
    plt.plot(Vloss, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Val Loss")
    plt.savefig("loss_curve.png")
    # acc curve
    plt.figure(figsize=(5, 5))
    plt.plot(Tacc, label='Train Acc')
    plt.plot(Vacc, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train and Val Accuraccy")
    plt.savefig("acc_curve.png")
    # LR curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epoch+1), lr_l3, marker='o', linestyle='-', label='Layer3 LR')
    plt.plot(range(1, num_epoch+1), lr_l4, marker='o', linestyle='-', label='Layer4 LR')
    plt.plot(range(1, num_epoch+1), lr_fc, marker='o', linestyle='-', label='FC Layer LR')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR Schedule")
    plt.grid(True)
    plt.savefig("LR_curve.png")
    

def Conf(val_labels, val_preds):
    cm = confusion_matrix(val_labels, val_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_normalized, annot=False, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
