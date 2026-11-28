from pathlib import Path
import json
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

ROOT = Path("datasets/processed/asl_alphabet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 5

device = torch.device("cpu")
print("Device:", device)

train_df = pd.read_csv(ROOT / "train.csv")
val_df = pd.read_csv(ROOT / "val.csv")

classes = sorted(train_df["label"].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}

with open(ROOT / "classes.json", "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

class ASLDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = class_to_idx[row["label"]]
        return image, label

class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

train_loader = DataLoader(
    ASLDataset(train_df), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    ASLDataset(val_df), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

model = ASLCNN(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0

print("\nASL classes:", len(classes))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("\nStarting training...\n")

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | "
        f"Val Accuracy: {accuracy:.2f}% | "
        f"Time: {(time.time()-start)/60:.1f} min"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE
        }, MODEL_DIR / "asl_alphabet_cnn.pth")
        print("Best model saved!")

print("\nTraining completed!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Model: models/asl_alphabet_cnn.pth")
