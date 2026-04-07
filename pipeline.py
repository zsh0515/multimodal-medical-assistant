import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"       # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"] = "1"        # 禁止 datasets 联网
os.environ["HF_HUB_OFFLINE"] = "1" 

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import ollama

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_STORE_DIR = "./models/faiss_index"
MODEL_DIR = "./models"
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────
# 1. 图像编码器
# ──────────────────────────────────────────
class ImageEncoder(nn.Module):
    def __init__(self, checkpoint_path, num_classes=7, output_dim=512):
        super().__init__()
        base = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        base.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Linear(2048, output_dim)

    def forward(self, x):
        feat = self.backbone(x).flatten(start_dim=1)
        return self.projector(feat)

# ──────────────────────────────────────────
# 2. 文本编码器
# ──────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, bert_dir, output_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert = AutoModel.from_pretrained(bert_dir)
        self.projector = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, texts):
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            out = self.bert(**encoded)
        return self.projector(out.last_hidden_state[:, 0, :])

# ──────────────────────────────────────────
# 3. Cross-Attention 融合
# ──────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, output_dim=256):
        super().__init__()
        self.img_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.txt_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
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
        img_out, _ = self.img_attn(img, txt, txt)
        img_out = self.norm_img(img + img_out).squeeze(1)
        txt_out, _ = self.txt_attn(txt, img, img)
        txt_out = self.norm_txt(txt + txt_out).squeeze(1)
        return self.classifier(torch.cat([img_out, txt_out], dim=-1))

# ──────────────────────────────────────────
# 4. RAG 检索器
# ──────────────────────────────────────────
class MedicalRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.store = FAISS.load_local(
            VECTOR_STORE_DIR, embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str) -> str:
        docs = self.store.similarity_search(query, k=self.top_k)
        parts = []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            parts.append(f"[Reference {i+1} - {src}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

# ──────────────────────────────────────────
# 5. 生成模型（FLAN-T5）
# ──────────────────────────────────────────
class DiagnosisGenerator:
    def __init__(self):
        self.model_name = "qwen2.5:3b"
        # 测试连接
        print(f"正在连接 Ollama ({self.model_name})...")
        try:
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "hello"}]
            )
            print("生成模型加载完成 ✓")
        except Exception as e:
            print(f"Ollama 连接失败，请确认已运行 ollama pull qwen2.5:3b\n错误: {e}")

    def generate(self, symptom_text: str, context: str) -> str:
        prompt = f"""你是一个皮肤科医疗助手。根据患者症状和参考医学知识，给出诊断建议。

患者症状：{symptom_text}

参考医学知识：
{context}

请用中文回答以下三点：
1. 最可能的皮肤病诊断是什么？
2. 判断依据是什么？
3. 建议患者下一步怎么做？"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

# ──────────────────────────────────────────
# 6. 完整 Pipeline
# ──────────────────────────────────────────
class MedicalPipeline:
    def __init__(self):
        print("\n正在初始化各模块...")

        self.img_encoder = ImageEncoder(
            checkpoint_path="./models/resnet50_skin.pth",
            num_classes=7,
            output_dim=512
        ).to(DEVICE)
        self.img_encoder.eval()
        print("  图像编码器 ✓")

        self.txt_encoder = TextEncoder(
            bert_dir=os.path.join(MODEL_DIR, "clinical_bert"),
            output_dim=512
        ).to(DEVICE)
        self.txt_encoder.eval()
        print("  文本编码器 ✓")

        self.fusion = CrossAttentionFusion(
            dim=512, num_heads=8, output_dim=256
        ).to(DEVICE)
        # 加载已保存的融合模块权重
        ckpt = torch.load(os.path.join(MODEL_DIR, "fusion.pth"), map_location=DEVICE)
        self.fusion.load_state_dict(ckpt["state_dict"])
        self.fusion.eval()
        print("  融合模块 ✓")

        self.retriever = MedicalRetriever(top_k=3)
        print("  RAG 检索器 ✓")

        self.generator = DiagnosisGenerator()

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        print("\n所有模块初始化完成，Pipeline 就绪 ✓\n")

    def run(self, image_path: str, symptom_text: str) -> dict:
        # Step 1: 图像特征
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.img_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = self.img_encoder(img_tensor)          # (1, 512)

        # Step 2: 文本特征
        with torch.no_grad():
            txt_feat = self.txt_encoder([symptom_text])      # (1, 512)

        # Step 3: 多模态融合
        with torch.no_grad():
            fused_feat = self.fusion(img_feat, txt_feat)     # (1, 256)

        # Step 4: RAG 检索
        context = self.retriever.retrieve(symptom_text)

        # Step 5: 生成诊断建议
        diagnosis = self.generator.generate(symptom_text, context)

        return {
            "symptom_input":  symptom_text,
            "fused_feature":  fused_feat.shape,
            "retrieved_refs": context,
            "diagnosis":      diagnosis,
        }


# ──────────────────────────────────────────
# 测试入口
# ──────────────────────────────────────────
if __name__ == "__main__":
    pipeline = MedicalPipeline()

    # 用数据集里任意一张图做测试
    test_cases = [
        {
            "image": "data/HAM10000_images_part_1/ISIC_0024306.jpg",
            "symptom": "左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。"
        },
        {
            "image": "data/HAM10000_images_part_1/ISIC_0024307.jpg",
            "symptom": "双侧肘部出现红色鳞状斑块，反复发作，伴有慢性瘙痒，无发热。"
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"案例 {i+1}")
        print(f"{'='*60}")
        print(f"症状描述: {case['symptom']}\n")

        result = pipeline.run(case["image"], case["symptom"])

        print(f"融合特征维度: {result['fused_feature']}")
        print(f"\n【检索到的参考文献】")
        print(result["retrieved_refs"])
        print(f"\n【诊断建议】")
        print(result["diagnosis"])

    print("\nPipeline 完整测试完成 ✓")