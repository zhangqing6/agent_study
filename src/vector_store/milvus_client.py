from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
import numpy as np
from typing import List, Dict, Any, Optional
import time


class MilvusClient:
    """Milvus 向量数据库客户端"""

    def __init__(self, config: dict):
        self.host = config.get("milvus_host", "localhost")
        self.port = config.get("milvus_port", "19530")
        self.collection_name = config.get("collection_name", "knowledge_base")
        self.dim = config.get("embedding_dim", 1024)

        # 连接 Milvus
        connections.connect(host=self.host, port=self.port)

        # 获取或创建集合
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合"""
        if utility.has_collection(self.collection_name):
            return Collection(self.collection_name)

        # 创建集合
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="create_time", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, "Knowledge Base")
        collection = Collection(self.collection_name, schema)

        # 创建索引
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)

        return collection

    def insert_documents(self, texts: List[str],
                         embeddings: Optional[List[np.ndarray]] = None,
                         metadatas: Optional[List[Dict]] = None) -> bool:
        """插入文档"""
        try:
            # 准备数据
            entities = [
                texts,  # text
                embeddings or [[]] * len(texts),  # embedding
                metadatas or [{}] * len(texts),  # metadata
                [int(time.time())] * len(texts)  # create_time
            ]

            self.collection.insert(entities)
            self.collection.flush()
            return True
        except Exception as e:
            print(f"插入失败: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索相似文档"""
        self.collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "text": hit.entity.get("text"),
                    "score": hit.score,
                    "metadata": hit.entity.get("metadata")
                })

        return formatted