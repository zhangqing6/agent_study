import os
from typing import List, Dict, Any
from .chunker import SemanticChunker
from ..vector_store.milvus_client import MilvusClient


class DataPipeline:
    """数据处理管道"""

    def __init__(self, config: dict):
        self.config = config
        self.chunker = SemanticChunker(config)
        self.milvus = MilvusClient(config)
        self.data_dir = config.get("data_dir", "./data/raw")

    def process_file(self, filepath: str, metadata: dict = None) -> bool:
        """处理单个文件"""
        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # 语义分块
            chunks = self.chunker.chunk_document(text)

            # 插入向量库
            success = self.milvus.insert_documents(
                texts=chunks,
                metadatas=[metadata or {"source": filepath}] * len(chunks)
            )

            return success
        except Exception as e:
            print(f"处理文件失败 {filepath}: {e}")
            return False

    def process_directory(self) -> Dict[str, Any]:
        """批量处理目录下的所有文档"""
        results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "chunks": 0
        }

        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.txt', '.pdf', '.md')):
                filepath = os.path.join(self.data_dir, filename)
                results["total"] += 1

                success = self.process_file(filepath, {"filename": filename})
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1

        return results