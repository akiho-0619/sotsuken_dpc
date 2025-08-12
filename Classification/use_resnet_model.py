import torch
from torchvision import models
import torch.nn as nn

num_classes = 15
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(
    torch.load(
        r"E:\sotsuken\Classification\model.pth", map_location=torch.device("cpu")
    )
)
model.eval()

from PIL import Image
from torchvision import transforms

# 画像の前処理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img = Image.open(r"E:\sotsuken\pictures\pic20250709.jpg")
input_tensor = transform(img).unsqueeze(0)  # バッチ次元追加

with torch.no_grad():
    outputs = model(input_tensor)
    print("出力:", outputs)
    _, predicted = torch.max(outputs, 1)
    print("予測クラスID:", predicted.item())
