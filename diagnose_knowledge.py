import os
os.environ["HF_HOME"] = "D:/model/hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KNOWLEDGE_DIR = "./knowledge_base"
VECTOR_STORE_DIR = "./models/faiss_index"

def load_all_documents():
    all_docs = []

    # 加载 txt 文件
    print("正在加载 txt 文档...")
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(KNOWLEDGE_DIR, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  ✓ {filename} ({len(docs)} 段)")

    # 加载 PDF 文件
    print("正在加载 PDF 文档...")
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(KNOWLEDGE_DIR, filename)
            print(f"  正在解析 {filename}，页数较多时需要等待...")
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  ✓ {filename} ({len(docs)} 页)")

    print(f"\n共加载 {len(all_docs)} 个文档块")
    return all_docs


def rebuild_vector_store(docs):
    print("\n正在切分文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # PDF 内容更丰富，适当加大块大小
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"切分为 {len(chunks)} 个文本块")

    print("\n正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("正在重建 FAISS 向量索引（文档较多时需要等待）...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"\n向量数据库已更新保存到 {VECTOR_STORE_DIR}/")
    return vector_store


def test_retrieval(vector_store):
    print("\n--- 检索测试 ---")
    queries = [
        "皮肤黑色素瘤的诊断标准",
        "湿疹的治疗方法",
        "皮肤病变的鉴别诊断",
    ]
    for q in queries:
        print(f"\n查询: {q}")
        results = vector_store.similarity_search(q, k=2)
        for i, doc in enumerate(results):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "")
            page_info = f" 第{page+1}页" if page != "" else ""
            print(f"  [{i+1}] 来源: {src}{page_info}")
            print(f"       内容: {doc.page_content[:100].strip()}...")


if __name__ == "__main__":
    docs = load_all_documents()
    vector_store = rebuild_vector_store(docs)
    test_retrieval(vector_store)
    print("\nPDF 知识库构建完成 ✓")