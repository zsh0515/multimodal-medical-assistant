import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import torch
import ollama
import jieba
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_DIR = "./models/faiss_index"

# ──────────────────────────────────────────
# 测试数据集：症状 + 标准参考回答
# ──────────────────────────────────────────
TEST_CASES = [
    {
        "symptom": "左臂有一块颜色不均匀的深色痣，边缘不规则，近3个月明显增大，偶尔瘙痒。",
        "reference": "该患者症状符合黑色素瘤的ABCDE特征，包括不对称、边缘不规则、颜色不均匀和直径变化。"
                     "建议立即就医进行皮肤镜检查，必要时进行活检以明确诊断。"
    },
    {
        "symptom": "双侧肘部出现红色鳞状斑块，反复发作，伴有慢性瘙痒，无发热。",
        "reference": "患者表现符合银屑病（牛皮癣）的典型症状，好发于肘部和膝部。"
                     "建议使用外用皮质类固醇治疗，同时避免诱发因素如压力和皮肤损伤。"
    },
    {
        "symptom": "老年患者背部有粗糙的棕色凸起斑点，无痛感，存在多年，近期无变化。",
        "reference": "描述符合脂溢性角化病的特征，这是一种常见的良性皮肤肿瘤。"
                     "通常不需要治疗，如有美观需求可考虑冷冻治疗或激光去除。"
    },
    {
        "symptom": "婴儿面颊部出现红色湿疹样皮疹，伴有渗液，瘙痒明显，有过敏性鼻炎家族史。",
        "reference": "症状符合特应性皮炎（湿疹）的诊断，与过敏体质密切相关。"
                     "建议保持皮肤湿润，使用温和保湿剂，急性期可短期使用外用激素。"
    },
    {
        "symptom": "面部出现珍珠样半透明小结节，边缘有毛细血管扩张，缓慢生长。",
        "reference": "该描述高度提示基底细胞癌，是最常见的皮肤恶性肿瘤之一。"
                     "建议及时就医，首选治疗方式为手术切除或Mohs显微描记手术。"
    },
]

# ──────────────────────────────────────────
# 初始化检索器
# ──────────────────────────────────────────
def init_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings,
        allow_dangerous_deserialization=True
    )
    return store

# ──────────────────────────────────────────
# 生成诊断
# ──────────────────────────────────────────
def generate_diagnosis(symptom: str, context: str) -> str:
    prompt = f"""你是一个皮肤科医疗助手。根据患者症状和参考医学知识，给出诊断建议。

患者症状：{symptom}

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
    return response["message"]["content"]

# ──────────────────────────────────────────
# 评估指标计算
# ──────────────────────────────────────────
def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    r1_list, r2_list, rl_list = [], [], []
    for pred, ref in zip(predictions, references):
        # 中文用结巴分词，词之间加空格，让 ROUGE 能正确匹配
        pred_cut = " ".join(jieba.cut(pred))
        ref_cut  = " ".join(jieba.cut(ref))
        scores = scorer.score(ref_cut, pred_cut)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
    return {
        "ROUGE-1": sum(r1_list) / len(r1_list),
        "ROUGE-2": sum(r2_list) / len(r2_list),
        "ROUGE-L": sum(rl_list) / len(rl_list),
    }


def compute_bert_score(predictions, references):
    # 临时关闭离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"

    P, R, F1 = bert_score(
        predictions, references,
        lang="zh",
        model_type="bert-base-chinese",
        num_layers=12,
        verbose=False
    )

    # 恢复离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    return {
        "BERTScore-P":  P.mean().item(),
        "BERTScore-R":  R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }

# ──────────────────────────────────────────
# 主评估流程
# ──────────────────────────────────────────
def main():
    print("正在初始化检索器...")
    store = init_retriever()
    print("检索器初始化完成 ✓\n")

    predictions = []
    references  = []
    results     = []

    for i, case in enumerate(TEST_CASES):
        print(f"正在评估案例 {i+1}/{len(TEST_CASES)}...")
        print(f"  症状: {case['symptom'][:40]}...")

        # 检索
        docs = store.similarity_search(case["symptom"], k=3)
        context = "\n\n".join([
            f"[参考{j+1}] {doc.page_content.strip()}"
            for j, doc in enumerate(docs)
        ])

        # 生成
        prediction = generate_diagnosis(case["symptom"], context)
        predictions.append(prediction)
        references.append(case["reference"])

        results.append({
            "case_id":    i + 1,
            "symptom":    case["symptom"],
            "reference":  case["reference"],
            "prediction": prediction,
        })

        print(f"  生成完成 ✓")

    # 计算指标
    print("\n正在计算评估指标...")
    rouge_scores = compute_rouge(predictions, references)
    bert_scores  = compute_bert_score(predictions, references)

    # 汇总结果
    print("\n" + "=" * 55)
    print("评估结果汇总")
    print("=" * 55)
    print(f"  ROUGE-1      : {rouge_scores['ROUGE-1']:.4f}")
    print(f"  ROUGE-2      : {rouge_scores['ROUGE-2']:.4f}")
    print(f"  ROUGE-L      : {rouge_scores['ROUGE-L']:.4f}")
    print(f"  BERTScore-P  : {bert_scores['BERTScore-P']:.4f}")
    print(f"  BERTScore-R  : {bert_scores['BERTScore-R']:.4f}")
    print(f"  BERTScore-F1 : {bert_scores['BERTScore-F1']:.4f}")
    print("=" * 55)

    # 保存详细结果
    os.makedirs("./results", exist_ok=True)
    output = {
        "metrics": {**rouge_scores, **bert_scores},
        "cases":   results
    }
    with open("./results/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n详细结果已保存到 ./results/evaluation.json ✓")
    print("评估完成 ✓")


if __name__ == "__main__":
    main()