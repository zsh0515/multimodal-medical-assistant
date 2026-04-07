import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"       # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"] = "1"        # 禁止 datasets 联网
os.environ["HF_HUB_OFFLINE"] = "1" 

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import ollama
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTOR_STORE_DIR = "./models/faiss_index"
MODEL_DIR = "./models"

# ──────────────────────────────────────────
# 各模块定义（与 pipeline.py 完全一致）
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
            parts.append(f"[参考文献 {i+1} - {src}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)


# ──────────────────────────────────────────
# 初始化 Pipeline（全局只加载一次）
# ──────────────────────────────────────────
print("正在初始化系统，请稍候...")

img_encoder = ImageEncoder(
    checkpoint_path="models/resnet50_skin.pth",
    num_classes=7, output_dim=512
).to(DEVICE)
img_encoder.eval()

txt_encoder = TextEncoder(
    bert_dir=os.path.join(MODEL_DIR, "clinical_bert"),
    output_dim=512
).to(DEVICE)
txt_encoder.eval()

fusion = CrossAttentionFusion(dim=512, num_heads=8, output_dim=256).to(DEVICE)
ckpt = torch.load(os.path.join(MODEL_DIR, "fusion.pth"), map_location=DEVICE)
fusion.load_state_dict(ckpt["state_dict"])
fusion.eval()

retriever = MedicalRetriever(top_k=3)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

print("系统初始化完成 ✓")

# ──────────────────────────────────────────
# 核心推理函数
# ──────────────────────────────────────────
def diagnose(image, symptom_text):
    if image is None:
        return "⚠️ 请上传皮肤图像", "", ""
    if not symptom_text.strip():
        return "⚠️ 请输入症状描述", "", ""

    try:
        # Step 1: 图像特征
        img = Image.fromarray(image).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = img_encoder(img_tensor)

        # Step 2: 文本特征
        with torch.no_grad():
            txt_feat = txt_encoder([symptom_text])

        # Step 3: 多模态融合
        with torch.no_grad():
            fused_feat = fusion(img_feat, txt_feat)

        # Step 4: RAG 检索
        context = retriever.retrieve(symptom_text)

        # Step 5: 生成诊断
        prompt = f"""你是一个皮肤科医疗助手。根据患者症状和参考医学知识，给出诊断建议。

患者症状：{symptom_text}

参考医学知识：
{context}

请用中文回答以下三点：
1. 最可能的皮肤病诊断是什么？
2. 判断依据是什么？
3. 建议患者下一步怎么做？"""

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}]
        )
        diagnosis = response["message"]["content"]

        return diagnosis, context, f"融合特征维度: {tuple(fused_feat.shape)}"

    except Exception as e:
        return f"❌ 系统错误: {str(e)}", "", ""


# ──────────────────────────────────────────
# Gradio 界面
# ──────────────────────────────────────────
with gr.Blocks(title="多模态医疗问诊助手") as demo:

    gr.Markdown("""
    # 🏥 多模态皮肤病问诊助手
    **上传皮肤图像 + 描述症状，系统将结合图像分析与医学知识库给出诊断建议。**
    > ⚠️ 本系统仅供学习研究使用，不构成真实医疗建议，请以医生诊断为准。
    """)

    with gr.Row():
        # 左栏：输入
        with gr.Column(scale=1):
            gr.Markdown("### 📋 输入信息")
            image_input = gr.Image(
                label="上传皮肤图像",
                type="numpy",
                height=300
            )
            symptom_input = gr.Textbox(
                label="症状描述",
                placeholder="例如：左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。",
                lines=4
            )
            submit_btn = gr.Button("🔍 开始诊断", variant="primary", size="lg")

        # 右栏：输出
        with gr.Column(scale=1):
            gr.Markdown("### 📊 诊断结果")
            diagnosis_output = gr.Textbox(
                label="诊断建议",
                lines=10,
                interactive=False
            )
            feature_output = gr.Textbox(
                label="模型信息",
                lines=1,
                interactive=False
            )

    with gr.Accordion("📚 检索到的参考文献", open=False):
        reference_output = gr.Textbox(
            label="RAG 检索结果",
            lines=8,
            interactive=False
        )

    # 示例输入
    gr.Examples(
        examples=[
            [None, "左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。"],
            [None, "双侧肘部出现红色鳞状斑块，反复发作，伴有慢性瘙痒，无发热。"],
            [None, "老年患者背部有粗糙的棕色凸起斑点，无痛感，存在多年。"],
        ],
        inputs=[image_input, symptom_input],
        label="示例输入"
    )

    submit_btn.click(
        fn=diagnose,
        inputs=[image_input, symptom_input],
        outputs=[diagnosis_output, reference_output, feature_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True       # 自动打开浏览器
    )