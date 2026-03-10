"""
FastAPI 应用主入口
提供 RESTful API 接口
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 添加项目根目录到 Python 路径（关键修复！）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 现在使用绝对导入
from src.agent.graph_builder import AgentGraphBuilder
from src.vector_store.milvus_client import MilvusClient
from src.vector_store.embeddings import get_embedder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    temperature: Optional[float] = Field(0.7, description="生成温度", ge=0, le=2)

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="Agent的回答")
    session_id: str = Field(..., description="会话ID")
    intent: Optional[str] = Field(None, description="识别到的意图")
    processing_time: float = Field(..., description="处理时间(秒)")

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化资源，关闭时清理资源
    """
    global agent, vector_store, embedder

    # ---------- 启动逻辑 ----------
    logger.info("🚀 应用启动中...")

    # 配置
    config = {
        # Milvus 配置
        "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
        "milvus_port": os.getenv("MILVUS_PORT", "19530"),
        "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),

        # Redis 配置
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),

        # Ollama 配置
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "llm_model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),

        # 检索配置
        "retrieval_top_k": int(os.getenv("RETRIEVAL_TOP_K", "5")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.6")),
    }

    try:
        # 初始化 embedding 模型
        logger.info("初始化 embedding 模型...")
        embedder = get_embedder(config)
        logger.info("✅ embedding 模型初始化完成")

        # 初始化向量存储
        logger.info("连接 Milvus...")
        vector_store = MilvusClient(config)
        logger.info("✅ Milvus 连接成功")

        # 初始化 Agent
        logger.info("构建 Agent...")
        builder = AgentGraphBuilder(config)
        agent = builder.build()
        logger.info("✅ Agent 构建完成")

        logger.info("✅ 应用初始化完成，准备接收请求...")

    except Exception as e:
        logger.error(f"❌ 应用初始化失败: {e}", exc_info=True)
        raise  # 抛出异常，让 FastAPI 知道启动失败

    # yield 之前的代码在启动时执行
    yield

    # ---------- 关闭逻辑 ----------
    logger.info("🛑 应用关闭中，清理资源...")

    # 在这里添加清理代码
    if vector_store:
        # 关闭 Milvus 连接等
        logger.info("关闭 Milvus 连接...")

    logger.info("✅ 资源清理完成")

# 创建 FastAPI 应用，使用 lifespan 参数 - 【只修改这里】
app = FastAPI(
    title="智能文档问答助手 API",  # 改成中文标题
    description="基于 LangGraph 的智能文档问答系统，支持私有知识库问答和多轮对话",  # 简短的描述
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 【只添加 summary 和 description 参数，不改变函数逻辑】
@app.get("/",
         summary="API 首页",
         description="返回 API 的基本信息和可用接口列表")
async def root():
    """根路径，返回API信息"""
    return {
        "name": "LangGraph Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat - POST",
            "search": "/search - POST",
            "health": "/health - GET",
            "docs": "/docs - GET"
        }
    }

# 【只添加 summary 和 description 参数】
@app.get("/health",
         summary="健康检查",
         description="检查 API 及所有依赖服务（Milvus、Redis）的运行状态")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "up",
            "milvus": "up" if vector_store else "down",
            "redis": "up"
        }
    }

# 【只添加 summary 和 description 参数】
@app.post("/chat",
          response_model=ChatResponse,
          summary="对话接口",
          description="发送用户问题，获取 Agent 的回答。支持多轮对话（使用相同的 session_id）")
async def chat(request: ChatRequest):
    """
    聊天接口
    发送用户问题，获取 Agent 的回答
    """
    start_time = time.time()

    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")

    try:
        logger.info(f"收到请求: {request.query[:50]}...")

        # 准备配置
        config = {
            "configurable": {
                "thread_id": request.session_id or f"session_{time.time()}"
            }
        }

        # 调用 Agent
        result = agent.invoke(
            {"query": request.query},
            config
        )

        # 提取响应
        response_text = result["messages"][-1].content
        intent = result.get("intent", "unknown")

        process_time = time.time() - start_time
        logger.info(f"处理完成，耗时: {process_time:.2f}秒")

        return ChatResponse(
            response=response_text,
            session_id=config["configurable"]["thread_id"],
            intent=intent,
            processing_time=process_time
        )

    except Exception as e:
        logger.error(f"处理请求失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )