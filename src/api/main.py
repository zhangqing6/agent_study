"""
FastAPI 应用主入口
提供 RESTful API 接口
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 新增导入
from fastapi.responses import FileResponse    # 新增导入
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

# 请求/响应模型 - 简化响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    temperature: Optional[float] = Field(0.7, description="生成温度", ge=0, le=2)

class ChatResponse(BaseModel):
    """聊天响应模型 - 只返回前端需要的字段"""
    response: str = Field(..., description="Agent的回答")
    session_id: str = Field(..., description="会话ID")

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

# 创建 FastAPI 应用，使用 lifespan 参数
app = FastAPI(
    title="智能文档问答助手",
    description="基于 LangGraph 的智能文档问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【新增】挂载静态文件目录
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"✅ 静态文件目录挂载: {static_path}")
else:
    logger.warning(f"⚠️ 静态文件目录不存在: {static_path}")

# 【修改】根路径现在返回漂亮的聊天界面
@app.get("/", include_in_schema=False)  # include_in_schema=False 表示不在API文档中显示
async def serve_chat():
    """提供聊天界面"""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {"message": "聊天界面文件不存在，请先创建 static/index.html"}

# 【保留】健康检查接口，但可以在API文档中隐藏
@app.get("/health", include_in_schema=True)
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口
    发送用户问题，获取 Agent 的回答
    """
    start_time = time.time()
    logger.info(f"=== 新请求 ===")
    logger.info(f"Session ID: {request.session_id}")
    logger.info(f"Query: {request.query}")

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
        logger.info(f"Thread ID: {config['configurable']['thread_id']}")

        # 调用 Agent
        result = agent.invoke(
            {"query": request.query},
            config
        )

        if "messages" in result:
            logger.info(f"返回的消息历史长度: {len(result['messages'])}")

        # 提取响应
        response_text = result["messages"][-1].content

        process_time = time.time() - start_time
        logger.info(f"处理完成，耗时: {process_time:.2f}秒")

        # 返回简化后的响应
        return ChatResponse(
            response=response_text,
            session_id=config["configurable"]["thread_id"]
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