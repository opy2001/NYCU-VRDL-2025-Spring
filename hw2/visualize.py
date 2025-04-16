import matplotlib.pyplot as plt

# =========== plot training curves ==================================
def plot_curve(save_dir, loss, lr, mAP):
    # ===== train loss =====
    plt.figure(figsize=(6,5))
    plt.plot(loss, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_dir}loss.png")
    # ===== LR =====
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(lr)+1), lr, marker='o', linestyle='-', label='LR')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.savefig(f"{save_dir}lr.png")
    # ===== valid mAP =====
    plt.figure(figsize=(6,5))
    plt.plot(mAP, label='Validation mAP')
    plt.xlabel("Epoch")
    plt.ylabel("AP @ IoU [0.5:0.95]")
    plt.legend()
    plt.title("Validation mAP")
    plt.savefig(f"{save_dir}mAP.png")