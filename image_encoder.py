import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./models"

class ImageEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int, output_dim=512):
        super().__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        base.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Linear(2048, output_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(start_dim=1)
        return self.projector(feat)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder = ImageEncoder(
        checkpoint_path="./models/resnet50_skin.pth",
        num_classes=7,
        output_dim=512
    ).to(DEVICE)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dummy_img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    img_tensor = transform(dummy_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(img_tensor)

    print(f"输出特征维度: {features.shape}")

    # 保存完整编码器（backbone + projector）
    save_path = os.path.join(SAVE_DIR, "image_encoder.pth")
    torch.save({
        "backbone": encoder.backbone.state_dict(),
        "projector": encoder.projector.state_dict(),
        "num_classes": 7,
        "output_dim": 512,
    }, save_path)

    print(f"模型已保存到 {SAVE_DIR}/")
    print("  └── image_encoder.pth   ← backbone + projector 权重")
    print("图像编码器运行正常 ✓")