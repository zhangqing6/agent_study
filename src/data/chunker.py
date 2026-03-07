import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class SemanticChunker:
    """语义分块器：基于语义相似度的文档分块"""

    def __init__(self, config: dict):
        self.model_name = config.get("embedding_model", "BAAI/bge-m3")
        self.embedder = SentenceTransformer(self.model_name)
        self.min_chunk_size = config.get("min_chunk_size", 3)  # 最小句子数
        self.max_chunk_size = config.get("max_chunk_size", 10)  # 最大句子数
        self.similarity_threshold = config.get("similarity_threshold", 0.7)

    def chunk_document(self, text: str) -> List[str]:
        """将文档分割成语义块"""
        # 分句
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_chunk_size:
            return [text]

        # 获取句子嵌入
        embeddings = self.embedder.encode(sentences)

        # 语义分块
        chunks = self._semantic_chunking(sentences, embeddings)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """中文分句"""
        # 按标点分割
        sentences = re.split(r'[。！？!?]', text)
        # 过滤空句子和太短的句子
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _semantic_chunking(self, sentences: List[str],
                           embeddings: np.ndarray) -> List[str]:
        """语义分块核心算法"""
        chunks = []
        current_chunk = [sentences[0]]
        current_embs = [embeddings[0]]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            emb = embeddings[i]

            # 计算与当前块的最大相似度
            similarities = [self._cosine_similarity(emb, ce) for ce in current_embs]
            max_sim = max(similarities)

            if max_sim >= self.similarity_threshold and len(current_chunk) < self.max_chunk_size:
                current_chunk.append(sentence)
                current_embs.append(emb)
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_embs = [emb]

        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))