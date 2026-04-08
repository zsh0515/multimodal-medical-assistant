# 多模态医疗问诊助手

基于 PyTorch + RAG + Transformer + CNN 的皮肤病智能问诊系统。

## 项目简介

本项目构建了一个融合图像与文本的多模态医疗问诊系统，
用户上传皮肤图像并描述症状，系统结合医学知识库给出诊断建议。

## 技术架构

用户输入（图像 + 症状文字）  
↓  
┌───────────────────────────────┐  
│  图像编码器（ResNet-50 + CNN） │  ← PyTorch + torchvision  
│  文本编码器（Clinical-BERT）   │  ← HuggingFace Transformers  
└───────────────┬───────────────┘  
↓  
Cross-Attention 融合模块      ← 多模态特征对齐（创新点）  
↓  
RAG 检索增强生成层           ← LangChain + FAISS  
（医学知识库 + 诊断学教材）  
↓  
大语言模型生成诊断建议        ← Ollama + Qwen2.5  
↓  
诊断报告输出  

## 核心技术栈

| 模块         | 技术                        |  
|--------------|-----------------------------|  
| 图像特征提取 | ResNet-50 迁移学习           |  
| 文本理解     | Clinical-BERT               |  
| 多模态融合   | Cross-Attention             |  
| 知识检索     | FAISS 向量数据库             |  
| 文本生成     | Qwen2.5-3B (Ollama本地推理) |  
| 知识库       | 诊断学教材 + 医学文献        |  
| 界面         | Gradio                      |  
| 评估指标     | ROUGE / BERTScore           |  

## 创新点

1. **跨模态动态注意力融合**：图像与文本特征通过
   Cross-Attention 互相感知，优于简单拼接方案
2. **RAG 知识溯源**：每条诊断建议附带检索来源，
   具备可审计性，降低大模型幻觉风险
3. **全本地化部署**：所有模型本地运行，无需联网，
   保护患者隐私

## 目录结构
skin-diagenose/  
├── data/                    # 训练数据集（HAM10000）  
├── knowledge_base/          # 医学知识文档  
├── models/                  # 所有模型权重  
│   ├── clinical_bert/  
│   ├── faiss_index/  
│   ├── resnet50_skin.pth  
│   ├── image_encoder.pth  
│   ├── text_projector.pth  
│   └── fusion.pth  
├── results/                 # 评估结果  
├── train.py                 # 图像模型训练  
├── image_encoder.py         # 图像编码器  
├── text_encoder.py          # 文本编码器  
├── fusion.py                # 多模态融合模块  
├── build_rag.py             # 构建向量数据库  
├── rag_retriever.py         # RAG 检索器  
├── add_pdf_knowledge.py     # PDF 知识库导入  
├── pipeline.py              # 完整推理流程  
├── app.py                   # Gradio 网页界面  
├── evaluate.py              # 评估脚本  
└── README.md  

## 快速启动

**1. 启动 Ollama**  
```bash  
ollama serve  
```  

**2. 启动网页界面**  
```bash  
python app.py  
```  

浏览器打开 http://127.0.0.1:7860

## 评估结果

| 指标          | 得分   |  
|---------------|--------|  
| ROUGE-1       | -      |  
| ROUGE-2       | -      |  
| ROUGE-L       | -      |  
| BERTScore-F1  | -      |  

> 运行 `python evaluate.py` 后填入实际结果

## 数据集

- 图像训练：[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
  （皮肤镜图像，7类皮肤病，10015张）  
- 医学知识库：诊断学教材 + 皮肤科医学文献  

## 环境依赖
```bash  
conda create -n medai python=3.10  
conda activate medai  
pip install torch torchvision transformers datasets  
pip install langchain langchain-community langchain-text-splitters  
pip install faiss-cpu sentence-transformers gradio  
pip install bert-score rouge-score pypdf ollama  
```  
