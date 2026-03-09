"""
文档添加脚本：将 data/raw/ 下的文档处理并存入 Milvus 知识库
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入项目中的模块
from src.data.pipeline import DataPipeline
from src.data.loader import get_loader, PDFLoader, TextLoader
from src.vector_store.embeddings import get_embedder
from src.vector_store.milvus_client import MilvusClient

def main():
    """主函数：处理 data/raw 下的所有文档并存入 Milvus"""
    logger.info("="*50)
    logger.info("开始文档处理流程")
    logger.info("="*50)

    # 1. 配置参数（和你的API服务保持一致）
    config = {
        "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
        "milvus_port": os.getenv("MILVUS_PORT", "19530"),
        "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "embedding_device": "cpu",
        "embedding_dim": 1024,
        "chunk_size": 500,          # 可根据需要调整
        "chunk_overlap": 50,         # 可根据需要调整
        "data_dir": "./src/data/raw"
    }

    # 2. 初始化核心组件
    logger.info("初始化 Embedding 模型...")
    embedder = get_embedder(config)

    logger.info("连接 Milvus...")
    milvus_client = MilvusClient(config)

    # 3. 获取所有待处理的文件
    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return

    # 支持的文件类型
    supported_extensions = ['.pdf', '.txt', '.md']
    files_to_process = []
    for ext in supported_extensions:
        files_to_process.extend(data_dir.glob(f"*{ext}"))

    if not files_to_process:
        logger.warning(f"在 {data_dir} 中没有找到支持的文档类型: {supported_extensions}")
        return

    logger.info(f"找到 {len(files_to_process)} 个待处理文件:")
    for f in files_to_process:
        logger.info(f"  - {f.name}")

    # 4. 逐个处理文件
    total_chunks = 0
    for file_path in files_to_process:
        logger.info(f"\n--- 处理文件: {file_path.name} ---")

        try:
            # 4.1 选择合适的加载器
            file_extension = file_path.suffix.lower()
            loader = get_loader(file_extension, config)

            # 4.2 加载文档
            logger.info("  加载文档...")
            documents = loader.load(str(file_path))
            if not documents:
                logger.warning("  文档加载失败或为空，跳过")
                continue

            # 4.3 初始化 Pipeline 并进行分块
            # 注意：你的 pipeline.py 里需要有 chunker 的实例
            pipeline = DataPipeline(config)
            all_chunks = []

            for doc in documents:
                text = doc.get("text", "")
                if not text:
                    continue
                # 使用语义分块器
                chunks = pipeline.chunker.chunk_document(text)
                logger.info(f"  生成 {len(chunks)} 个文本块")
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning("  没有生成任何文本块，跳过")
                continue

            # 4.4 生成 embeddings
            logger.info("  生成向量 embeddings...")
            embeddings_array = embedder.embed_documents(all_chunks)  # 返回 ndarray (n, dim)

            # 4.5 准备元数据
            metadatas = [{"source": file_path.name, "chunk_id": i} for i in range(len(all_chunks))]

            # 4.6 存入 Milvus - 将 ndarray 转换为 list of ndarray
            logger.info("  存入 Milvus...")
            # 将二维数组转换为列表，每个元素是一个一维向量
            embeddings_list = [embeddings_array[i] for i in range(len(embeddings_array))]

            success = milvus_client.insert_documents(
                texts=all_chunks,
                embeddings=embeddings_list,
                metadatas=metadatas
            )

            if success:
                logger.info(f"  ✅ 文件 {file_path.name} 处理成功，添加 {len(all_chunks)} 个块")
                total_chunks += len(all_chunks)
            else:
                logger.error(f"  ❌ 文件 {file_path.name} 存入 Milvus 失败")

        except Exception as e:
            logger.error(f"处理文件 {file_path.name} 时发生错误: {e}", exc_info=True)

    logger.info("\n" + "="*50)
    logger.info(f"处理完成！总共添加 {total_chunks} 个文档块到知识库")
    logger.info("="*50)

if __name__ == "__main__":
    main()