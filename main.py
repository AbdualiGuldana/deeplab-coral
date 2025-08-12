import os
import json
import time
import datetime
from tqdm import tqdm


import pandas as pd 

import seaborn as sns
import torch
from torch import nn, Tensor
from torch.optim import SGD, Adam
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config.model_cfg import *
from models import get_model
from utils import ImgClsDataset
from utils import get_transforms
from models.loss import DiceFocalLoss
from models.metrics import meanIoU
from util import base_collate_fn
from config.model_cfg import DEVICE
from torch.optim.lr_scheduler import StepLR


CLASS_NAMES = [
    "background",  # class 0
    "dry",         # class 1
    "humid",       # class 2
    "slush",       # class 3
    "snow",        # class 4
    "wet"          # class 5
]

# class_weights = torch.tensor([
#     1.40,   # background
#     17.57,  # dry
#     9.93,   # humid
#     40.32,  # slush
#     277.78, # snow
#     10.15   # wet
# ], device=DEVICE)

def get_loss_for_loss(losses: dict):
    log_losses = dict()
    for key, val in losses.items():
        log_losses[key] = val.detach().cpu().item()
    
    return log_losses


# def calculate_iou(pred, target, num_classes):
#     """
#     각 클래스에 대해 IoU를 계산하고 mIoU를 반환합니다.

#     Args:
#         pred (torch.Tensor): 예측 값 (B, H, W) 형태. 예측된 클래스 인덱스를 포함.
#         target (torch.Tensor): 실제 값 (B, H, W) 형태. 실제 클래스 인덱스를 포함.
#         num_classes (int): 클래스 수.

#     Returns:
#         list: 각 클래스별 IoU
#         float: mIoU
#     """
#     ious = []
#     pred = pred.view(-1)  # (B * H * W,)
#     target = target.view(-1)  # (B * H * W,)

#     for cls in range(num_classes):
#         # 배경 연산 제외
#         if cls == 0:
#             continue
#         # 각 클래스에 대한 Intersection과 Union 계산
#         intersection = ((pred == cls) & (target == cls)).sum().item()
#         union = ((pred == cls) | (target == cls)).sum().item()

#         if union == 0:
#             ious.append(float('nan'))  # 해당 클래스가 없으면 IoU는 NaN 처리
#         else:
#             ious.append(intersection / union)

#     # mIoU 계산 (NaN 값 제외)
#     valid_ious = [iou for iou in ious if not np.isnan(iou)]
#     miou = np.mean(valid_ious) if valid_ious else float('nan')

#     return ious, miou


def model_load(model):
    start_epoch, _ = os.path.splitext(os.path.basename(MODEL_CFG.load_from))
    chkp = torch.load(MODEL_CFG.load_from)
    model.load_state_dict(chkp, strict=False)
    
    print(f"{MODEL_CFG.load_from} is load.")
    return model, int(start_epoch)


log_path = "output/train_log.txt"
os.makedirs("output", exist_ok=True)

# Clear old log file (optional)
with open(log_path, "w") as f:
    f.write("epoch,phase,batch,total_batch,loss,time,avg_miou,is_best\n")

def get_confusion_matrix(pred, label, num_classes):
        mask = (label >= 0) & (label < num_classes)
        conf = np.bincount(
            num_classes * label[mask].astype(int) + pred[mask].astype(int),
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return conf

def compute_metrics(conf: np.ndarray):
    diag = np.diag(conf).astype(np.float64)
    rows = conf.sum(axis=1).astype(np.float64)
    cols = conf.sum(axis=0).astype(np.float64)
    total = conf.sum().astype(np.float64)

    pixel_acc = diag.sum() / total if total > 0 else 0.0

    union = rows + cols - diag
    iou = np.where(union > 0, diag / union, 0.0)
    mean_iou = iou.mean()

    dice = np.where(rows + cols > 0, 2 * diag / (rows + cols), 0.0)
    mean_dice = dice.mean()

    freq = rows / total if total > 0 else np.zeros_like(rows)
    freq_weighted_iou = (freq * iou).sum()

    return pixel_acc, mean_iou, mean_dice, freq_weighted_iou, iou, dice


early_stopping_patience = 8
epochs_without_improvement = 0


def train():
    lr_history = []
    os.makedirs("output", exist_ok=True)
    best_miou = 0.0
    train_anno_path = os.path.join(DATA_CFG.root_path, DATA_CFG.train_anno_path)
    with open(train_anno_path, 'r') as rf:
        train_anno = json.load(rf)

    transforms_lst = []
    for config in TRAIN_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)
    
    train_dataset = ImgClsDataset(DATA_CFG, 
                                  train_anno,
                                  transforms=transforms_lst
                                  )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=TRAIN_DATALOADER_CFG.bathc_size,
                                  shuffle=TRAIN_DATALOADER_CFG.shuffle,
                                  num_workers=TRAIN_DATALOADER_CFG.num_worker,
                                  collate_fn=base_collate_fn)

    valid_anno_path = os.path.join(
        DATA_CFG.root_path,
        DATA_CFG.valid_anno_path
    )

    with open(valid_anno_path, 'r') as rf:
        valid_anno = json.load(rf)

    transforms_lst = []
    for config in VALID_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)

    valid_dataset = ImgClsDataset(
        DATA_CFG, 
        valid_anno,
        transforms=transforms_lst
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_DATALOADER_CFG.bathc_size,
        shuffle=VALID_DATALOADER_CFG.shuffle,
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    model = get_model(MODEL_TYPE, MODEL_CFG)
    model = model.to(DEVICE)
    
   # loss_fn = nn.CrossEntropyLoss() # nn.MSELoss(reduction='mean') # DiceFocalLoss()
    #loss_fn = DiceFocalLoss()
    loss_fn = DiceFocalLoss(dice_weight=0.7, ce_weight=0.3)

    #dan changed lr to 1e-4 from 1e-5
    optim = Adam(model.parameters(), lr=1e-5)
    #scheduler = StepLR(optim, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=15, T_mult=2, eta_min=1e-6)
    num_batch = len(train_dataloader)
    digits = len(str(num_batch))

    total_epoch = TRAIN_CFG.num_epoch
    start_epoch = 0
    if MODEL_CFG.load_from is not None:
        model, start_epoch = model_load(model)

    remaining_iters = len(train_dataloader) * (TRAIN_CFG.num_epoch - start_epoch + 1)

    for epoch in range(start_epoch, total_epoch+1):
        batch_cnt = 0
        log_losses = list()
        model = model.train()
        for images, masks, sensors, annotations in train_dataloader:
            batch_cnt += 1
            start_time = time.time()
            # images = images
            outs = model(model, images=images, sensors=sensors)['out'].squeeze()
            losses = dict(
                loss=loss_fn(outs, masks.long())
            )

            log_losses.append(get_loss_for_loss(losses))
            losses: Tensor = sum(losses.values())
            losses.backward()
            if batch_cnt % TRAIN_CFG.accum_step == 0:
                optim.step()
                optim.zero_grad()
                

            iter_time = time.time() - start_time
            log_losses[-1].update(time=iter_time)

            print(f"Epoch {epoch}/{total_epoch} | Batch {batch_cnt}/{num_batch} | loss={log_losses[-1]['loss']:.6f} | lr={optim.param_groups[0]['lr']:.2e} | {iter_time:.3f}s", flush=True)

            if batch_cnt % TRAIN_CFG.log_step == 0 or batch_cnt == len(train_dataloader):
                now = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
                log_lsg = f"[{now}] Epoch(Train) [{epoch}/{total_epoch}][{batch_cnt: {digits}}/{num_batch}] "
                print(log_lsg, flush=True)
                with open(log_path, "a") as f:
                    f.write(f"{epoch},train,{batch_cnt},{num_batch},{log_losses[-1]['loss']:.6f},{log_losses[-1]['time']:.4f},,\n")

                    
        torch.save(model.state_dict(), f"output/{epoch}.pth")

        scheduler.step()
        for param_group in optim.param_groups:
                    current_lr = param_group['lr']
                    lr_history.append(current_lr)
                    print(f"🔁 Learning Rate: {current_lr}")
        # miou_lst = []
        # cls_iou_lst = []

        # model = model.eval()

        # with torch.no_grad():
        #     for images, masks, sensors, _ in valid_dataloader:
        #         outs = model(model, images, sensors)['out'].squeeze().detach().cpu()
                
        #         _, preds = torch.max(outs, 1)

        #         ious, miou = calculate_iou(preds, masks.detach().cpu(), 6)
        #         cls_iou_lst.append(ious)
        #         miou_lst.append(miou)

        # #print(f"[{now}] Epoch(Valid) [{epoch}/{total_epoch}] mIoU: {sum(miou_lst) / len(miou_lst):.4f}")
        # avg_miou = sum(miou_lst) / len(miou_lst)

        conf_matrix = np.zeros((6, 6), dtype=np.int64)

        with torch.no_grad():
            for images, masks, sensors, _ in valid_dataloader:
                outs = model(model, images, sensors)['out'].squeeze().detach().cpu()

                if outs.ndim == 4:
                    preds = outs.argmax(dim=1)
                else:
                    preds = outs

                labels = masks.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()

                for p, l in zip(preds_np, labels):
                    mask = (l >= 0) & (l < 6)
                    conf_matrix += np.bincount(
                        6 * l[mask].astype(int) + p[mask].astype(int),
                        minlength=6 ** 2
                    ).reshape(6, 6)

        # Now compute metrics
        diag = np.diag(conf_matrix).astype(np.float64)
        rows = conf_matrix.sum(axis=1).astype(np.float64)
        cols = conf_matrix.sum(axis=0).astype(np.float64)
        union = rows + cols - diag
        iou = np.where(union > 0, diag / union, 0.0)
        avg_miou = iou.mean()

        pixel_acc, mean_iou, mean_dice, freq_w_iou, iou_per_class, dice_per_class = compute_metrics(conf_matrix)

        print(f"[{now}] Epoch(Valid) [{epoch}/{total_epoch}] mIoU: {avg_miou:.4f}")

        is_best = avg_miou > best_miou
        print(f"Pixel Acc: {pixel_acc:.4f} | Mean Dice: {mean_dice:.4f}")
        if is_best:
            best_miou = avg_miou
            epochs_without_improvement = 0  # reset on improvement
            torch.save(model.state_dict(), f"output/best.pth")
            torch.save(model.state_dict(), f"output/best_epoch_{epoch}_miou_{avg_miou:.4f}.pth")
            print(f"✅ Best model updated at epoch {epoch} with mIoU {best_miou:.4f}")
        else:
            epochs_without_improvement += 1


        with open(log_path, "a") as f:
            f.write(f"{epoch},valid,,,,,{avg_miou:.4f},{'yes' if is_best else 'no'}\n")

        with open("output/miou_dice_per_epoch.csv", "a") as f:
            if epoch == start_epoch:
                f.write("epoch,mean_iou,mean_dice\n")
            f.write(f"{epoch},{mean_iou:.4f},{mean_dice:.4f}\n")

        #Visualize 
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(lr_history, marker='o')
        plt.title("Learning Rate per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.savefig("output/learning_rate_schedule.png")
        plt.close()

        with open("output/learning_rate_log.txt", "w") as f:
            for epoch, lr in enumerate(lr_history, start=1):
                f.write(f"{epoch},{lr}\n")

       

        # Plot mIoU and Dice over epochs
        df = pd.read_csv("output/miou_dice_per_epoch.csv")
        
        plt.figure()
        plt.plot(df["epoch"], df["mean_iou"], marker='o', label="Mean IoU")
        plt.plot(df["epoch"], df["mean_dice"], marker='s', label="Mean Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("mIoU & Dice Over Epochs")
        plt.legend()
        plt.grid(True)
        for x, y in zip(df["epoch"], df["mean_iou"]):
            y = float(y)
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

        for x, y in zip(df["epoch"], df["mean_dice"]):
            y = float(y)
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

       
        plt.tight_layout()
        plt.savefig("output/miou_dice_trend.png")
        plt.close()

        if epochs_without_improvement >= early_stopping_patience:
            print(f"⏹️ Early stopping triggered: No improvement in {early_stopping_patience} epochs.")
            break




def valid():
    def get_confusion_matrix(pred, label, num_classes):
        mask = (label >= 0) & (label < num_classes)
        conf = np.bincount(
            num_classes * label[mask].astype(int) + pred[mask].astype(int),
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return conf

    valid_anno_path = os.path.join(
        DATA_CFG.root_path,
        DATA_CFG.valid_anno_path
    )

    with open(valid_anno_path, 'r') as rf:
        valid_anno = json.load(rf)

    transforms_lst = []
    for config in VALID_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)

    valid_dataset = ImgClsDataset(
        DATA_CFG,
        valid_anno,
        transforms=transforms_lst
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_DATALOADER_CFG.bathc_size,
        shuffle=VALID_DATALOADER_CFG.shuffle,
        num_workers=VALID_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    model = get_model(MODEL_TYPE, MODEL_CFG)
    if MODEL_CFG.load_from is not None:
        model, _ = model_load(model)

    model = model.to(DEVICE)
    inference_times = []
    memory_usages = []
    # miou_lst = []

    conf_matrix = np.zeros((6, 6), dtype=np.int64) 

    for images, masks, sensors, _ in tqdm(valid_dataloader):
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            outs = model(model, images, sensors)['out'].squeeze().detach().cpu()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # ms
        max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

        inference_times.append(elapsed_time)
        memory_usages.append(max_mem)
        # If outs has shape [B, C, H, W], convert to [B, H, W] by taking argmax over channel dim
        if outs.ndim == 4 and outs.shape[1] > 1:
            preds = outs.argmax(dim=1)
        else:
            preds = outs
        labels = masks.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        for p, l in zip(preds_np, labels):
            conf_matrix += get_confusion_matrix(p, l, num_classes=6)

        #miou = meanIoU(preds, masks.detach().cpu())
        # _, miou = calculate_iou(preds, masks.detach().cpu(), num_classes=6)

        # if miou is not None:
        #     miou_lst.append(miou)
        # else:
        #     miou_lst.append(0)  # or skip, or handle as you wish

        print(f"Inference time: {elapsed_time:.2f} ms, Max GPU memory: {max_mem:.2f} MB")
    


    # print(f"mIoU: {sum(miou_lst) / len(miou_lst):.4f}")
    # print(miou_lst)
    pixel_acc, mean_iou, mean_dice, freq_w_iou, iou_per_class, dice_per_class = compute_metrics(conf_matrix)

    print(f"\nFinal Validation Metrics:")
    print(f"Pixel Accuracy:           {pixel_acc:.4f}")
    print(f"Mean IoU:                 {mean_iou:.4f}")
    print(f"Mean Dice:                {mean_dice:.4f}")
    print(f"Frequency-Weighted IoU:   {freq_w_iou:.4f}")
    print("\nPer-class IoU and Dice:")
    for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
        print(f"{i}: {CLASS_NAMES[i]:<12} | IoU: {iou:.4f} | Dice: {dice:.4f}")


    with open("output/validation_metrics.txt", "w") as f:
        f.write(f"Pixel Accuracy:         {pixel_acc:.4f}\n")
        f.write(f"Mean IoU:               {mean_iou:.4f}\n")
        f.write(f"Mean Dice:              {mean_dice:.4f}\n")
        f.write(f"Freq Weighted IoU:      {freq_w_iou:.4f}\n")
        f.write(f"IoU per class:          {iou_per_class}\n")
        f.write(f"Dice per class:         {dice_per_class}\n")

    plt.figure(figsize=(8, 4))
    x = np.arange(len(iou_per_class))
    bar_width = 0.4

    # Bar plots
    bars1 = plt.bar(x - bar_width/2, iou_per_class, width=bar_width, alpha=0.6, label="IoU")
    bars2 = plt.bar(x + bar_width/2, dice_per_class, width=bar_width, alpha=0.6, label="Dice")


    # Labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    plt.xlabel("Class Index")
    plt.ylabel("Score")
    plt.title("Per-Class IoU & Dice")
    # Show mean IoU and Dice on the plot
    plt.text(0.99, 0.95, f"Mean IoU: {mean_iou:.4f}", transform=plt.gca().transAxes,
            fontsize=10, ha='right', va='top', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.text(0.99, 0.88, f"Mean Dice: {mean_dice:.4f}", transform=plt.gca().transAxes,
            fontsize=10, ha='right', va='top', color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.xticks(x, CLASS_NAMES, rotation=30, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/per_class_metrics.png")
    plt.close()


    # Normalize by row (i.e., ground truth)
    conf_norm = conf_matrix.astype(np.float64) / conf_matrix.sum(axis=1, keepdims=True)
    conf_norm = np.nan_to_num(conf_norm)  # Replace NaN with 0 for rows with no instances

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        cbar_kws={'label': 'Prediction Proportion'}
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png")
    plt.close()

    

    # Visualization
    plt.figure(figsize=(12, 4))

    # --- Inference Time ---

    plt.subplot(1, 3, 1)
    plt.plot(inference_times, marker='o')
    plt.title('Inference Time per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Time (ms)')
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    plt.axhline(avg_time, color='red', linestyle='--', label=f'Avg: {avg_time:.2f} ms')
    plt.axhline(min_time, color='green', linestyle=':', label=f'Min: {min_time:.2f} ms')
    plt.axhline(max_time, color='orange', linestyle=':', label=f'Max: {max_time:.2f} ms')
    plt.legend()
    plt.text(0.99, 0.01, f"Avg: {avg_time:.2f}\nMin: {min_time:.2f}\nMax: {max_time:.2f}",
             transform=plt.gca().transAxes, fontsize=10, va='bottom', ha='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # --- GPU Memory ---
    plt.subplot(1, 3, 2)
    plt.plot(memory_usages, marker='o', color='orange')
    plt.title('Max GPU Memory per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Memory (MB)')
    avg_mem = np.mean(memory_usages)
    min_mem = np.min(memory_usages)
    max_mem = np.max(memory_usages)
    plt.axhline(avg_mem, color='red', linestyle='--', label=f'Avg: {avg_mem:.2f} MB')
    plt.axhline(min_mem, color='green', linestyle=':', label=f'Min: {min_mem:.2f} MB')
    plt.axhline(max_mem, color='blue', linestyle=':', label=f'Max: {max_mem:.2f} MB')
    plt.legend()
    plt.text(0.99, 0.01, f"Avg: {avg_mem:.2f}\nMin: {min_mem:.2f}\nMax: {max_mem:.2f}",
             transform=plt.gca().transAxes, fontsize=10, va='bottom', ha='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # # --- mIoU ---
    # plt.subplot(1, 3, 3)
    # plt.plot(miou_lst, marker='o', color='green')
    # plt.title('mIoU per Batch')
    # plt.xlabel('Batch')
    # plt.ylabel('mIoU')
    # avg_miou = np.mean(miou_lst)
    # min_miou = np.min(miou_lst)
    # max_miou = np.max(miou_lst)
    # plt.axhline(avg_miou, color='red', linestyle='--', label=f'Avg: {avg_miou:.4f}')
    # plt.axhline(min_miou, color='green', linestyle=':', label=f'Min: {min_miou:.4f}')
    # plt.axhline(max_miou, color='blue', linestyle=':', label=f'Max: {max_miou:.4f}')
    # plt.legend()
    # plt.text(0.99, 0.01, f"Avg: {avg_miou:.4f}\nMin: {min_miou:.4f}\nMax: {max_miou:.4f}",
    #          transform=plt.gca().transAxes, fontsize=10, va='bottom', ha='right',
    #          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # plt.tight_layout()
    # os.makedirs('output', exist_ok=True)
    # plt.savefig('output/validation_performance.png')
   
    #plt.show()



def test():
    model = get_model(MODEL_TYPE, MODEL_CFG)
    if MODEL_CFG.load_from is not None:
        model, _ = model_load(model)
    model = model.to(DEVICE)
    model.eval()

    test_anno_path = os.path.join(DATA_CFG.root_path, DATA_CFG.test_anno_path)
    with open(test_anno_path, 'r') as rf:
        test_anno = json.load(rf)

    transforms_lst = []
    for config in TEST_PIPE:
        transforms_lst.append(get_transforms(config))
    transforms_lst = Compose(transforms_lst)

    test_dataset = ImgClsDataset(DATA_CFG, test_anno, transforms=transforms_lst)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_DATALOADER_CFG.bathc_size,
        shuffle=TEST_DATALOADER_CFG.shuffle,
        num_workers=TEST_DATALOADER_CFG.num_worker,
        collate_fn=base_collate_fn
    )

    os.makedirs("output/test_vis", exist_ok=True)

    conf_matrix = np.zeros((6, 6), dtype=np.int64)
    img_count = 0

    for images, masks, sensors, annotations in tqdm(test_dataloader, desc="Testing"):
        with torch.no_grad():
            outs = model(model, images=images, sensors=sensors)['out'].squeeze().detach().cpu()

        if outs.ndim == 4:
            preds = outs.argmax(dim=1)
        else:
            preds = outs

        labels = masks.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        for idx in range(len(preds_np)):
            pred_mask = preds_np[idx]
            true_mask = labels[idx]
            image = images[idx].detach().cpu().numpy().transpose(1, 2, 0)

            # Update confusion matrix
            mask = (true_mask >= 0) & (true_mask < 6)
            conf_matrix += np.bincount(
                6 * true_mask[mask].astype(int) + pred_mask[mask].astype(int),
                minlength=6**2
            ).reshape(6, 6)

            # Save image
            fig, arr = plt.subplots(1, 3, figsize=(12, 4))
            arr[0].imshow(image)
            arr[0].set_title("Image")
            arr[1].imshow(pred_mask)
            arr[1].set_title("Predicted Mask")
            arr[2].imshow(true_mask)
            arr[2].set_title("Ground Truth")
            for ax in arr:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"output/test_vis/test.png", dpi=200)
            plt.close()
            img_count += 1

    # Compute final metrics
    pixel_acc, mean_iou, mean_dice, freq_w_iou, iou_per_class, dice_per_class = compute_metrics(conf_matrix)

    print(f"\n🔍 Final Test Metrics:")
    print(f"Pixel Accuracy:           {pixel_acc:.4f}")
    print(f"Mean IoU:                 {mean_iou:.4f}")
    print(f"Mean Dice:                {mean_dice:.4f}")
    print(f"Frequency-Weighted IoU:   {freq_w_iou:.4f}")
    print("\nPer-class IoU and Dice:")
    for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
        print(f"{i}: {CLASS_NAMES[i]:<12} | IoU: {iou:.4f} | Dice: {dice:.4f}")

    with open("output/test_metrics.txt", "w") as f:
        f.write(f"Pixel Accuracy:         {pixel_acc:.4f}\n")
        f.write(f"Mean IoU:               {mean_iou:.4f}\n")
        f.write(f"Mean Dice:              {mean_dice:.4f}\n")
        f.write(f"Freq Weighted IoU:      {freq_w_iou:.4f}\n")
        f.write(f"IoU per class:          {iou_per_class}\n")
        f.write(f"Dice per class:         {dice_per_class}\n")


if __name__ == "__main__":
    train()


