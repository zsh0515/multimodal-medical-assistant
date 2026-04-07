import os
os.environ["HF_HOME"] = "D:/model/hub"

from transformers import AutoTokenizer, AutoModel

print("正在下载 bert-base-chinese（约 400MB）...")
AutoTokenizer.from_pretrained("bert-base-chinese")
AutoModel.from_pretrained("bert-base-chinese")
print("下载完成 ✓")