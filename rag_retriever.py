import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"       # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"] = "1"        # 禁止 datasets 联网
os.environ["HF_HUB_OFFLINE"] = "1"  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_STORE_DIR = "./models/faiss_index"

class MedicalRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        print("正在加载向量数据库...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.store = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("向量数据库加载完成 ✓")

    def retrieve(self, query: str) -> str:
        """
        输入查询文本，返回拼接好的检索结果字符串，直接可注入 prompt。
        """
        docs = self.store.similarity_search(query, k=self.top_k)
        if not docs:
            return "No relevant medical knowledge found."

        parts = []
        for i, doc in enumerate(docs):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            parts.append(f"[Reference {i+1} - {src}]\n{doc.page_content.strip()}")

        return "\n\n".join(parts)


if __name__ == "__main__":
    retriever = MedicalRetriever(top_k=3)

    test_cases = [
        "Patient reports dark mole with irregular border and color changes",
        "Chronic itchy red patches on elbows and knees, scaly appearance",
        "Elderly patient has rough brown raised spots on chest",
    ]

    for query in test_cases:
        print(f"\n{'='*55}")
        print(f"症状输入: {query}")
        print(f"{'='*55}")
        context = retriever.retrieve(query)
        print(context)

    print("\nRAG 检索器封装完成 ✓")