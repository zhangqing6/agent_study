LangGraph Agent - 智能文档问答系统
基于 LangGraph 和 RAG架构 的智能文档问答系统，支持私有知识库检索、多轮对话记忆、本地大模型推理，提供完整的 API 服务与容器化部署方案。
```
演示视频：【LangGraph Agent - 智能文档问答系统】 https://www.bilibili.com/video/BV1i3wgzkEY7/?share_source=copy_web&vd_source=44cb484ede030bd06d3f1307adebee80
```
✨ 核心特性
🤖 AI Agent 智能决策：基于 LangGraph 实现意图识别-知识检索-答案生成三节点状态图，Agent 可自主判断调用知识库还是直接对话

🧠 多轮对话记忆：通过 LangGraph Checkpointer 实现对话历史持久化，支持跨轮次上下文理解

📚 私有知识库：支持 PDF/TXT/MD 文档上传，基于语义分块与向量检索（Milvus）实现精准问答

🖥️ 本地模型推理：集成 Ollama 本地部署 Qwen2.5:7B，无需联网，保障数据隐私

🐳 一键部署：Docker Compose 编排 6 个服务（Milvus、Redis、Ollama 等），开箱即用

📊 可视化文档：FastAPI + Swagger UI 提供交互式 API 文档，支持在线调试

🏗️ 系统架构


```
用户请求 → FastAPI → LangGraph Agent
                          ├─ 意图识别节点 → (rag_query/chat)
                          ├─ 知识检索节点 → Milvus向量库
                          └─ 答案生成节点 → Ollama(Qwen2.5:7B)
```

# 📁 项目结构

```
langgraph-agent/
├── src/
│   ├── agent/               # Agent核心模块
│   │   ├── graph_builder.py # LangGraph图构建
│   │   ├── state.py         # 状态定义
│   │   └── nodes/           # 功能节点
│   │       ├── intent.py    # 意图识别
│   │       ├── retrieval.py # 知识检索
│   │       └── generation.py # 答案生成
│   ├── data/                # 数据处理模块
│   │   ├── pipeline.py      # 数据处理Pipeline
│   │   ├── chunker.py       # 语义分块器
│   │   └── loader.py        # 文档加载器
│   ├── vector_store/        # 向量存储模块
│   │   ├── milvus_client.py # Milvus客户端
│   │   └── embeddings.py    # Embedding封装
│   └── api/                 # API服务模块
│       └── main.py          # FastAPI入口
├── docker-compose.yml        # Docker服务编排
├── manage.py                 # 一键管理脚本
├── add_docs.py               # 文档添加脚本
├── requirements.txt          # Python依赖
└── .env                      # 环境变量配置
   ```

# 前置要求
Python 3.10+

Docker & Docker Compose（用于 Milvus/Redis 等依赖服务）

Ollama（用于本地模型推理）

## 1. 安装依赖

### 克隆项目
```
git clone <https://github.com/zhangqing6/agent_study.git>
cd langgraph-agent
```
### 创建虚拟环境
```
python -m venv venv

source venv/bin/activate  # Linux/Mac

 或 .\venv\Scripts\activate  # Windows
```
### 安装依赖
```
pip install -r requirements.txt
```
## 2. 启动依赖服务



启动 Milvus、Redis 等
```
docker-compose up -d
```
查看服务状态
```
docker-compose ps
```
## 3. 启动 Ollama 模型服务

### 下载模型（首次需下载，约4.7GB）
```
ollama pull qwen2.5:7b
```
### 启动 Ollama 服务（指定端口）
```
export OLLAMA_HOST=127.0.0.1:11435  # Linux/Mac

 或 $env:OLLAMA_HOST = "127.0.0.1:11435"  # Windows

ollama serve
```
## 4. 启动 API 服务

### 新开终端
```
cd langgraph-agent

source venv/bin/activate

python -m src.api.main
```
访问 API 文档：http://localhost:8000/docs

或者http://localhost:8000前端界面

## 5. 一键启动
```
python manage.py start
```
### 📚 添加知识库文档

将文档放入 src/data/raw/ 目录，

然后运行：
```
python add_docs.py
```
支持格式：.txt、.pdf、.md

### 🧪 API 测试示例
#### 健康检查
```
curl http://localhost:8000/health
```
#### 聊天对话

```
curl -X POST "http://localhost:8000/chat" \

  -H "Content-Type: application/json" \

  -d '{
    "query": "公司的年假制度是什么？",
    "session_id": "user123"
  }'
```
#### 多轮对话测试
```
第1轮：{"query": "我叫小明", "session_id": "test1"}

第2轮：{"query": "我叫什么名字", "session_id": "test1"}

第3轮：{"query": "我刚才问了什么", "session_id": "test1"}
```
### 🔧 配置说明

#### 环境变量 (.env)

 ##### Milvus配置
```
MILVUS_HOST=localhost

MILVUS_PORT=19530

COLLECTION_NAME=knowledge_base
```
 ##### Redis配置
```
REDIS_HOST=localhost

REDIS_PORT=6379
```
 ##### Ollama配置
```
OLLAMA_BASE_URL=http://localhost:11435

LLM_MODEL=qwen2.5:7b

EMBEDDING_MODEL=BAAI/bge-m3
```
#### 端口说明
```
服务	        端口	      说明
API 服务 	8000	 主服务入口
Ollama   	11435	 LLM 推理服务
Milvus	    19530    向量数据库
Redis	    6379	 缓存服务
```
### 🛠️ 管理命令

####  一键启动所有服务
```
python manage.py start
```
#### 查看服务状态
```
python manage.py status
```
#### 添加文档到知识库
```
python manage.py add-docs
```
#### 查看知识库统计
```
python manage.py list-docs
```
#### 停止所有服务
```
python manage.py stop
```
### 🎯 核心功能实现
功能	实现方式	代码位置

意图识别	LangChain + Qwen2.5	src/agent/nodes/intent.py

知识检索	Milvus + BGE Embedding	src/agent/nodes/retrieval.py

答案生成	LangChain + Ollama	src/agent/nodes/generation.py

记忆功能	LangGraph MemorySaver	src/agent/graph_builder.py

文档处理	语义分块 + 向量化	src/data/pipeline.py
### ⚠️ 常见问题

Q: Ollama 端口冲突怎么办？

A: 设置环境变量 OLLAMA_HOST=127.0.0.1:11435，并修改代码中的 base_url。

Q: 模型下载失败？

A: 设置 HuggingFace 镜像源 export HF_ENDPOINT=https://hf-mirror.com

Q: Docker 服务无法启动？

A: 检查端口占用：netstat -ano | findstr :19530，释放被占用的端口。

### 📌 版本要求
Python 3.10+

Docker 20.10+

Ollama 0.1.0+

内存建议 8GB+（运行 7B 模型）

### 📄 许可证
MIT License

### 🙏 致谢
LangGraph

FastAPI

Milvus

Ollama

Qwen