"""
FastAPI 应用主入口
提供 RESTful API 接口
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入项目模块
from ..agent.graph_builder import AgentGraphBuilder
from ..vector_store.milvus_client import MilvusClient
from ..vector_store.embeddings import get_embedder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="LangGraph Agent API",
    description="基于 LangGraph 的智能文档问答 Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class DocumentRequest(BaseModel):
    """文档处理请求模型"""
    file_path: str = Field(..., description="文件路径")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索关键词")
    top_k: Optional[int] = Field(5, description="返回结果数量", ge=1, le=20)


class SearchResponse(BaseModel):
    """搜索响应模型"""
    results: List[Dict[str, Any]] = Field(..., description="搜索结果")
    total: int = Field(..., description="结果总数")


# 全局变量
agent = None
vector_store = None
embedder = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global agent, vector_store, embedder

    logger.info("正在初始化应用...")

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

        # 初始化向量存储
        logger.info("连接 Milvus...")
        vector_store = MilvusClient(config)

        # 初始化 Agent
        logger.info("构建 Agent...")
        builder = AgentGraphBuilder(config)
        agent = builder.build()

        logger.info("✅ 应用初始化完成")
    except Exception as e:
        logger.error(f"❌ 应用初始化失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    logger.info("正在关闭应用...")
    # 可以在这里添加清理代码
    logger.info("✅ 应用已关闭")


@app.get("/")
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


@app.get("/health")
async def health_check():
    """健康检查接口"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "up",
            "milvus": "unknown",
            "redis": "unknown"
        }
    }

    # 检查 Milvus 连接
    try:
        if vector_store and vector_store.collection:
            vector_store.collection.load()
            health_status["services"]["milvus"] = "up"
    except:
        health_status["services"]["milvus"] = "down"
        health_status["status"] = "degraded"

    return health_status


@app.post("/chat", response_model=ChatResponse)
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
        logger.error(f"处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    知识库搜索接口

    在向量库中搜索相关文档
    """
    if not vector_store or not embedder:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    try:
        # 生成查询向量
        query_embedding = embedder.embed_query(request.query)

        # 搜索
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k
        )

        return SearchResponse(
            results=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_document(request: DocumentRequest):
    """
    添加文档到知识库
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="向量存储未初始化")

    try:
        # 这里应该调用数据处理 pipeline
        # 暂时返回成功
        return {"status": "success", "message": "文档已添加到处理队列"}

    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    获取系统监控指标
    """
    return {
        "requests": {
            "total": 0,  # 这里应该从计数器获取
        },
        "system": {
            "timestamp": time.time()
        }
    }


# 中间件：请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求的中间件"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.3f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )