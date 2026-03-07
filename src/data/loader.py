"""
文档加载器模块
支持多种格式的文档加载
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """文档加载器基类"""

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config.get("data_dir", "./data/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, source: str) -> List[Dict[str, Any]]:
        """
        加载文档

        Args:
            source: 文档路径或URL

        Returns:
            文档列表，每个文档包含 text 和 metadata
        """
        raise NotImplementedError("子类必须实现 load 方法")


class TextLoader(DocumentLoader):
    """文本文件加载器"""

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """加载文本文件"""
        try:
            path = Path(filepath)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            return [{
                "text": text,
                "metadata": {
                    "source": str(path),
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "file_type": path.suffix
                }
            }]
        except Exception as e:
            logger.error(f"加载文本文件失败 {filepath}: {e}")
            return []


class PDFLoader(DocumentLoader):
    """PDF文件加载器"""

    def load(self, filepath: str) -> List[Dict[str, Any]]:
        """加载PDF文件"""
        try:
            # 这里需要安装 pypdf 或 pymupdf
            # 示例代码，实际使用时需要导入相应库
            import pypdf

            path = Path(filepath)
            documents = []

            with open(path, 'rb') as f:
                pdf = pypdf.PdfReader(f)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()

            return [{
                "text": text,
                "metadata": {
                    "source": str(path),
                    "filename": path.name,
                    "pages": len(pdf.pages),
                    "file_size": path.stat().st_size
                }
            }]
        except ImportError:
            logger.error("请安装 pypdf: pip install pypdf")
            return []
        except Exception as e:
            logger.error(f"加载PDF文件失败 {filepath}: {e}")
            return []


class WebLoader(DocumentLoader):
    """网页加载器"""

    def load(self, url: str) -> List[Dict[str, Any]]:
        """加载网页内容"""
        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # 移除 script 和 style 标签
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()

            text = soup.get_text()

            return [{
                "text": text,
                "metadata": {
                    "source": url,
                    "title": soup.title.string if soup.title else "",
                    "content_type": response.headers.get('content-type', '')
                }
            }]
        except Exception as e:
            logger.error(f"加载网页失败 {url}: {e}")
            return []


def get_loader(file_type: str, config: dict) -> DocumentLoader:
    """获取对应的加载器"""
    loaders = {
        '.txt': TextLoader,
        '.pdf': PDFLoader,
        '.md': TextLoader,
        'web': WebLoader
    }

    loader_class = loaders.get(file_type, TextLoader)
    return loader_class(config)