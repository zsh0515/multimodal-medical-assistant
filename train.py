import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

DATA_DIRS = ["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"]
CSV_PATH  = "data/HAM10000_metadata.csv"
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_map = {}
        for d in DATA_DIRS:
            for f in os.listdir(d):
                self.path_map[f.replace(".jpg", "")] = os.path.join(d, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.path_map[row["image_id"]]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

def main():
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(CSV_PATH)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["dx"])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    train_df, val_df = train_test_split(df, test_size=0.2,
                                        stratify=df["label"], random_state=42)
    train_ds = SkinDataset(train_df, train_tf)
    val_ds   = SkinDataset(val_df,   val_tf)

    # Windows 下 num_workers 必须为 0，否则报错
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), "./models/resnet50_skin.pth")
    print("Model saved → resnet50_skin.pth")

# Windows 多进程必须有这一行
if __name__ == "__main__":
    main()