import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"       # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"] = "1"        # 禁止 datasets 联网
os.environ["HF_HUB_OFFLINE"] = "1"  
import pickle
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KNOWLEDGE_DIR = "./knowledge_base"
VECTOR_STORE_DIR = "./models/faiss_index"

def build_vector_store():
    print("正在加载知识库文档...")
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"  加载了 {len(docs)} 个文档")

    print("正在切分文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  切分为 {len(chunks)} 个文本块")

    print("正在加载 Embedding 模型（首次需下载，约 90MB）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("正在构建 FAISS 向量索引...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"向量数据库已保存到 {VECTOR_STORE_DIR}/")
    print("  ├── index.faiss   ← 向量索引")
    print("  └── index.pkl     ← 文档元数据")
    return vector_store, embeddings


def test_retrieval(vector_store):
    print("\n--- 检索测试 ---")
    queries = [
        "dark irregular mole with color variation",
        "red scaly patches itching on elbows",
        "small brown raised growth on elderly patient",
    ]
    for q in queries:
        print(f"\n查询: {q}")
        results = vector_store.similarity_search(q, k=2)
        for i, doc in enumerate(results):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            print(f"  [{i+1}] 来源: {src}")
            print(f"       内容: {doc.page_content[:120].strip()}...")


if __name__ == "__main__":
    vector_store, embeddings = build_vector_store()
    test_retrieval(vector_store)
    print("\nRAG 知识库构建完成 ✓")