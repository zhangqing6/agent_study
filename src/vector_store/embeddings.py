"""
Embedding 模型封装模块
支持多种 embedding 模型，提供统一的接口
"""

import os  # 添加 os 模块导入
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging

# 设置 Hugging Face 镜像源（在文件最前面设置，确保最先执行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding 模型封装类"""

    def __init__(self, config: dict):
        """
        初始化 embedding 模型

        Args:
            config: 配置字典，包含：
                - embedding_model: 模型名称，如 "BAAI/bge-m3"
                - embedding_device: 设备，如 "cpu" 或 "cuda"
                - embedding_dim: 向量维度
                - embedding_batch_size: 批处理大小
        """
        # 再次确保环境变量设置（双重保险，防止被覆盖）
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        self.model_name = config.get("embedding_model", "BAAI/bge-m3")
        self.device = config.get("embedding_device", "cpu")
        self.dim = config.get("embedding_dim", 1024)  # bge-m3 默认 1024 维
        self.batch_size = config.get("embedding_batch_size", 32)

        logger.info(f"正在加载 embedding 模型: {self.model_name}")
        logger.info(f"使用 Hugging Face 镜像源: {os.environ.get('HF_ENDPOINT')}")

        # 加载模型时会自动使用镜像源
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"模型加载完成，向量维度: {self.dim}")

    def embed_query(self, text: str) -> np.ndarray:
        """
        为单个查询文本生成 embedding

        Args:
            text: 输入文本

        Returns:
            numpy 数组，shape (dim,)
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        为多个文档生成 embeddings

        Args:
            texts: 文本列表

        Returns:
            numpy 数组，shape (len(texts), dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        embed_documents 的别名，保持接口一致
        """
        return self.embed_documents(texts)


# 全局函数，方便调用
def get_embedder(config: dict) -> EmbeddingModel:
    """获取 embedding 模型实例"""
    return EmbeddingModel(config)


def get_default_embedder() -> EmbeddingModel:
    """使用默认配置获取 embedding 模型"""
    config = {
        "embedding_model": "BAAI/bge-m3",
        "embedding_device": "cpu",
        "embedding_dim": 1024
    }
    return EmbeddingModel(config)