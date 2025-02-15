from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "quickstart"

# 既存のインデックスをリストして確認
existing_indexes = pc.list_indexes()
print("Existing indexes:", existing_indexes)

# インデックスが存在しない場合のみ作成
if index_name not in [index.name for index in existing_indexes]:
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Index '{index_name}' already exists")

# インデックスの取得
index = pc.Index(index_name)

# インデックス情報の確認
print("Current indexes:", pc.list_indexes())
