import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"       # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"] = "1"        # 禁止 datasets 联网
os.environ["HF_HUB_OFFLINE"] = "1"  
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
SAVE_DIR = "./models"

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_dim = self.bert.config.hidden_size
        self.projector = nn.Linear(hidden_dim, output_dim)

    def forward(self, texts: list[str]):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            output = self.bert(**encoded)
        cls_vec = output.last_hidden_state[:, 0, :]
        return self.projector(cls_vec)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder = TextEncoder(output_dim=512).to(DEVICE)
    encoder.eval()

    test_symptoms = [
        "Patient has a dark irregular mole on the left arm, itching for 3 weeks.",
        "Red scaly patches on both elbows, no pain reported.",
        "Small brown flat spot on the back, stable for years.",
    ]

    with torch.no_grad():
        features = encoder(test_symptoms)

    print(f"输入: {len(test_symptoms)} 条症状描述")
    print(f"输出特征维度: {features.shape}")

    # 保存 projector 权重 + tokenizer + bert
    torch.save(encoder.projector.state_dict(),
               os.path.join(SAVE_DIR, "text_projector.pth"))
    encoder.bert.save_pretrained(os.path.join(SAVE_DIR, "clinical_bert"))
    encoder.tokenizer.save_pretrained(os.path.join(SAVE_DIR, "clinical_bert"))

    print(f"模型已保存到 {SAVE_DIR}/")
    print("  ├── clinical_bert/        ← BERT 权重 + tokenizer")
    print("  └── text_projector.pth   ← 投影层权重")
    print("文本编码器运行正常 ✓")