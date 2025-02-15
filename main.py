import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from pydantic import BaseModel, Field

# FastAPIの初期化
app = FastAPI(
    title="Pinecone Vector Search API",
    description="Vector similarity search using Pinecone",
    version="1.0.0",
)

# CORSの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 環境変数とPineconeの設定
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("quickstart")


# データモデル
class SearchQuery(BaseModel):
    query_vector: List[float] = Field(
        ..., example=[0.1, 0.2], description="Query vector for similarity search"
    )


class SearchResponse(BaseModel):
    matches: List[Dict[str, Any]] = Field(
        default=[], description="Matching vectors with similarity scores"
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <title>Pinecone ベクトル検索 API</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50">
        <div class="container mx-auto px-4 py-8 max-w-4xl">
            <header class="mb-8">
                <h1 class="text-3xl font-bold mb-4">Pinecone ベクトル検索 API</h1>
                <p class="text-gray-600 mb-2">ベクトル類似度検索のためのシンプルなAPIサービス</p>
            </header>

            <section class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4">コンセプト</h2>
                <p class="text-gray-700 mb-4">
                    このAPIは、Pineconeを活用した高性能なベクトル類似度検索を提供します。
                    テキスト埋め込み、画像特徴量、あるいは任意の数値ベクトルに対して、
                    高速で正確な類似度検索を実現します。
                </p>
            </section>

            <section class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4">主な機能</h2>
                <ul class="space-y-3 text-gray-700">
                    <li class="flex items-start">
                        <span class="text-blue-500 mr-2">→</span>
                        <span>ベクトルデータの追加・更新（<code class="bg-gray-100 px-1 rounded">/upsert</code>）</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-blue-500 mr-2">→</span>
                        <span>類似ベクトル検索（<code class="bg-gray-100 px-1 rounded">/search</code>）</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-blue-500 mr-2">→</span>
                        <span>インデックス状態の確認（<code class="bg-gray-100 px-1 rounded">/status</code>）</span>
                    </li>
                </ul>
            </section>

            <section class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4">利用シーン</h2>
                <ul class="space-y-3 text-gray-700">
                    <li class="flex items-start">
                        <span class="text-green-500 mr-2">•</span>
                        <span>類似文書検索システム</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-500 mr-2">•</span>
                        <span>画像類似度検索</span>
                    </li>
                    <li class="flex items-start">
                        <span class="text-green-500 mr-2">•</span>
                        <span>レコメンデーションシステム</span>
                    </li>
                </ul>
            </section>

            <div class="text-center">
                <a href="/docs" 
                   class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg inline-block transition duration-200">
                    API ドキュメントを見る（Swagger UI）
                </a>
            </div>

            <footer class="mt-8 text-center text-gray-500 text-sm">
                <p>※ 詳細な使用方法やパラメータについては、APIドキュメントをご確認ください。</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# StatusエンドポイントとIndexStats定義
class IndexStats(BaseModel):
    total_vector_count: int = Field(default=0)
    dimension: int = Field(default=2)
    namespaces: Dict[str, Any] = Field(default_factory=dict)


@app.get("/status")
async def status():
    try:
        indexes = pc.list_indexes()
        return {
            "indexes": [
                {"name": idx.name, "host": idx.host, "dimension": idx.dimension}
                for idx in indexes
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upsertエンドポイント
@app.post("/upsert")
async def upsert_vectors():
    try:
        vectors = [
            {
                "id": "vec1",
                "values": [0.1, 0.2],
                "metadata": {"description": "First vector"},
            },
            {
                "id": "vec2",
                "values": [0.2, 0.3],
                "metadata": {"description": "Second vector"},
            },
            {
                "id": "vec3",
                "values": [0.1, 0.15],
                "metadata": {"description": "Third vector"},
            },
        ]

        index.upsert(vectors=vectors)
        return {"message": "Vectors upserted successfully", "count": len(vectors)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 検索エンドポイント
@app.post(
    "/search",
    response_model=SearchResponse,
    description="Search for similar vectors",
    responses={
        200: {
            "description": "Successful search response",
            "content": {
                "application/json": {
                    "example": {
                        "matches": [
                            {
                                "id": "vec1",
                                "score": 0.98,
                                "metadata": {"description": "First vector"},
                            }
                        ]
                    }
                }
            },
        }
    },
)
async def search(query: SearchQuery):
    try:
        results = index.query(vector=query.query_vector, top_k=5, include_metadata=True)
        return {
            "matches": [
                {"id": match.id, "score": match.score, "metadata": match.metadata}
                for match in results.matches
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# カスタムSwagger UIエンドポイント
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Vector Search API - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
