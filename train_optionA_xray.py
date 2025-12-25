import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
DATA_DIR = "data_raw/chest_xray"   # change if needed
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
NUM_WORKERS = 2
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loaders():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_ds, train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(dev)
        logits = model(x).squeeze(1)          # shape [B]
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    probs = np.concatenate(probs).reshape(-1)
    ys = np.concatenate(ys).reshape(-1)

    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    yhat = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, yhat, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-9)   # recall for pneumonia class
    specificity = tn / (tn + fp + 1e-9)

    return {
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def main():
    dev = device()
    print("Device:", dev)

    train_ds, train_loader, val_loader, test_loader = make_loaders()
    print("Class mapping:", train_ds.class_to_idx)  # usually {'NORMAL':0, 'PNEUMONIA':1}

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary logit
    model = model.to(dev)

    # Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_auc = -1.0
    best_path = os.path.join(OUT_DIR, "best_resnet18_xray.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(dev)
            y = y.float().to(dev)  # 0/1

            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        val = evaluate(model, val_loader, dev)
        print("VAL:", val)

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best -> {best_path}")

    # Test best
    model.load_state_dict(torch.load(best_path, map_location=dev))
    test = evaluate(model, test_loader, dev)
    print("TEST:", test)

if __name__ == "__main__":
    main()
