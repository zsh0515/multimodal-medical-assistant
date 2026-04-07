import os
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./models"

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, output_dim=256):
        super().__init__()
        self.img_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.txt_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm_img = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, img_feat, txt_feat):
        img = img_feat.unsqueeze(1)
        txt = txt_feat.unsqueeze(1)

        img_out, _ = self.img_attn(query=img, key=txt, value=txt)
        img_out = self.norm_img(img + img_out).squeeze(1)

        txt_out, _ = self.txt_attn(query=txt, key=img, value=img)
        txt_out = self.norm_txt(txt + txt_out).squeeze(1)

        fused = torch.cat([img_out, txt_out], dim=-1)
        return self.classifier(fused)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    fusion = CrossAttentionFusion(dim=512, num_heads=8, output_dim=256).to(DEVICE)

    img_feat = torch.randn(4, 512).to(DEVICE)
    txt_feat = torch.randn(4, 512).to(DEVICE)

    out = fusion(img_feat, txt_feat)
    print(f"融合输出维度: {out.shape}")

    # 保存融合模块权重 + 配置
    save_path = os.path.join(SAVE_DIR, "fusion.pth")
    torch.save({
        "state_dict": fusion.state_dict(),
        "config": {
            "dim": 512,
            "num_heads": 8,
            "output_dim": 256,
        }
    }, save_path)

    print(f"模型已保存到 {SAVE_DIR}/")
    print("  └── fusion.pth   ← Cross-Attention 融合模块权重 + 配置")
    print("融合模块运行正常 ✓")