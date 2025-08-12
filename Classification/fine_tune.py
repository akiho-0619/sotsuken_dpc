import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

# CUDAが使用可能か確認
print(f"CUDA available: {torch.cuda.is_available()}")
print(
    f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)
# ハイパーパラメータ
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
num_classes = 15  # クラス数に応じて変更

# データセットのパス
# train_dir = r'E:\sotsuken\Classification\Vegetable Images\train'
# val_dir = r'E:\sotsuken\Classification\Vegetable Images\test'

# データ変換
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# データセットとデータローダ
image_datasets = {
    "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
    "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
}
dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
    "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False),
}

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロードとファインチューニング設定
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # テストデータでの正答率計算
        model.eval()
        test_corrects = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
                test_total += labels.size(0)
        test_acc = test_corrects.double() / test_total
        print(f"Test Accuracy: {test_acc:.4f}")
print("Training complete.")
