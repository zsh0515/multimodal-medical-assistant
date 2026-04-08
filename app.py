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
# 自定义 CSS 样式
# ──────────────────────────────────────────
custom_css = """
/* 全局背景 */
.gradio-container {
    background: #0a0a0f !important;
    min-height: 100vh;
    font-family: 'Segoe UI', sans-serif;
}

/* 粒子背景画布 */
#particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    pointer-events: none;
}

/* 主容器 */
.main-wrapper {
    position: relative;
    z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
}

/* 标题区域 */
.hero-section {
    text-align: center;
    padding: 3rem 0 2rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    from { filter: drop-shadow(0 0 20px rgba(102,126,234,0.3)); }
    to   { filter: drop-shadow(0 0 40px rgba(240,147,251,0.5)); }
}

.hero-subtitle {
    color: #8892a4;
    font-size: 1rem;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

.hero-badge {
    display: inline-flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}

.badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid;
}

.badge-purple { 
    background: rgba(102,126,234,0.15); 
    color: #667eea; 
    border-color: rgba(102,126,234,0.3); 
}
.badge-pink   { 
    background: rgba(240,147,251,0.15); 
    color: #f093fb; 
    border-color: rgba(240,147,251,0.3); 
}
.badge-teal   { 
    background: rgba(79,209,197,0.15);  
    color: #4fd1c5; 
    border-color: rgba(79,209,197,0.3); 
}
.badge-orange { 
    background: rgba(251,176,64,0.15);  
    color: #fbb040; 
    border-color: rgba(251,176,64,0.3); 
}

/* 卡片样式 */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent, rgba(102,126,234,0.5), transparent);
}

.glass-card:hover {
    border-color: rgba(102,126,234,0.3);
    background: rgba(255,255,255,0.05);
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3),
                0 0 30px rgba(102,126,234,0.1);
}

/* 输入区标题 */
.section-label {
    color: #a0aec0;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: linear-gradient(180deg, #667eea, #f093fb);
    border-radius: 2px;
}

/* Gradio 组件覆盖样式 */
.gradio-container .gr-form,
.gradio-container .gr-panel {
    background: transparent !important;
    border: none !important;
}

label.svelte-1b6s6xi {
    color: #8892a4 !important;
    font-size: 0.85rem !important;
}

.gradio-container textarea,
.gradio-container input[type=text] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s !important;
}

.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: rgba(102,126,234,0.5) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.1) !important;
    outline: none !important;
}

/* 诊断按钮 */
.diagnose-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    position: relative !important;
    overflow: hidden !important;
}

.diagnose-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(102,126,234,0.4) !important;
}

.diagnose-btn:active {
    transform: translateY(0) !important;
}

/* 结果输出区 */
.gradio-container .gr-box {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* 图片上传区 */
.gradio-container .upload-container {
    border: 2px dashed rgba(102,126,234,0.3) !important;
    border-radius: 12px !important;
    background: rgba(102,126,234,0.05) !important;
    transition: all 0.3s !important;
}

.gradio-container .upload-container:hover {
    border-color: rgba(102,126,234,0.6) !important;
    background: rgba(102,126,234,0.08) !important;
}

/* 免责声明 */
.disclaimer {
    text-align: center;
    color: #4a5568;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* 状态指示器 */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #48bb78;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* 折叠区域 */
.gradio-container .gr-accordion {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

/* 示例区域 */
.gradio-container .gr-samples {
    background: transparent !important;
}

/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { 
    background: rgba(102,126,234,0.3); 
    border-radius: 3px; 
}
"""

# ──────────────────────────────────────────
# 粒子背景 + 打字机效果 JS
# ──────────────────────────────────────────
particle_js = """
<canvas id="particle-canvas"></canvas>
<script>
(function() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animId;

    function resize() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    class Particle {
        constructor() { this.reset(); }
        reset() {
            this.x     = Math.random() * canvas.width;
            this.y     = Math.random() * canvas.height;
            this.vx    = (Math.random() - 0.5) * 0.4;
            this.vy    = (Math.random() - 0.5) * 0.4;
            this.r     = Math.random() * 1.5 + 0.5;
            this.alpha = Math.random() * 0.4 + 0.1;
            const colors = ['102,126,234','240,147,251','79,209,197','251,176,64'];
            this.color = colors[Math.floor(Math.random() * colors.length)];
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > canvas.width ||
                this.y < 0 || this.y > canvas.height) this.reset();
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${this.color},${this.alpha})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < 80; i++) particles.push(new Particle());

    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(102,126,234,${0.15 * (1 - dist/120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => { p.update(); p.draw(); });
        drawConnections();
        animId = requestAnimationFrame(animate);
    }
    animate();
})();
</script>
"""

# ──────────────────────────────────────────
# Gradio 界面
# ──────────────────────────────────────────
with gr.Blocks(
    title="多模态医疗问诊助手",
    css=custom_css
) as demo:

    # 粒子背景
    gr.HTML(particle_js)

    # 标题区
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-title">🏥 多模态皮肤病问诊助手</div>
        <div class="hero-subtitle">
            Multimodal Medical Skin Disease Diagnosis Assistant
        </div>
        <div class="hero-badge">
            <span class="badge badge-purple">CNN · ResNet-50</span>
            <span class="badge badge-pink">Transformer · BERT</span>
            <span class="badge badge-teal">RAG · FAISS</span>
            <span class="badge badge-orange">Qwen2.5 · Ollama</span>
        </div>
        <div style="margin-top:1rem; color:#4a5568; font-size:0.8rem;">
            <span class="status-dot"></span>系统就绪 · 本地推理 · 隐私保护
        </div>
    </div>
    """)

    with gr.Row():
        # 左栏：输入
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">输入信息</div>')
            image_input = gr.Image(
                label="上传皮肤图像",
                type="numpy",
                height=260
            )
            symptom_input = gr.Textbox(
                label="症状描述",
                placeholder="请详细描述症状，例如：左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。",
                lines=4
            )
            submit_btn = gr.Button(
                "🔍  开始诊断",
                variant="primary",
                elem_classes=["diagnose-btn"]
            )

        # 右栏：输出
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">诊断结果</div>')
            diagnosis_output = gr.Textbox(
                label="AI 诊断建议",
                lines=12,
                interactive=False,
                placeholder="诊断结果将在这里显示..."
            )
            feature_output = gr.Textbox(
                label="模型运行信息",
                lines=1,
                interactive=False
            )

    with gr.Accordion("📚 查看检索到的医学参考文献", open=False):
        reference_output = gr.Textbox(
            label="RAG 检索结果",
            lines=8,
            interactive=False
        )

    gr.Examples(
        examples=[
            [None, "左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。"],
            [None, "双侧肘部出现红色鳞状斑块，反复发作，伴有慢性瘙痒，无发热。"],
            [None, "老年患者背部有粗糙的棕色凸起斑点，无痛感，存在多年。"],
        ],
        inputs=[image_input, symptom_input],
        label="示例输入（点击填充）"
    )

    gr.HTML("""
    <div class="disclaimer">
        ⚠️ 本系统仅供学习研究使用，不构成真实医疗建议，请以专业医生诊断为准
        <br>基于 PyTorch · HuggingFace · LangChain · Ollama 构建 · 完全本地运行
    </div>
    """)

    submit_btn.click(
        fn=diagnose,
        inputs=[image_input, symptom_input],
        outputs=[diagnosis_output, reference_output, feature_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )