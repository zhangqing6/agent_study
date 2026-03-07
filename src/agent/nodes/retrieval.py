from ...vector_store.milvus_client import MilvusClient
from ...vector_store.embeddings import get_embedder


class RetrievalNode:
    """知识检索节点：从向量库获取相关知识"""

    def __init__(self, config: dict):
        self.milvus = MilvusClient(config)
        self.embedder = get_embedder(config)
        self.top_k = config.get("retrieval_top_k", 3)

    def __call__(self, state: AgentState) -> dict:
        query = state["query"]

        # 生成查询向量
        query_embedding = self.embedder.embed_query(query)

        # 检索相似文档
        results = self.milvus.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        docs = [r["text"] for r in results]
        scores = [r["score"] for r in results]

        return {
            "retrieved_docs": docs,
            "doc_scores": scores,
            "need_rag": len(docs) > 0,
            "steps": ["document_retrieval"]
        }