from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]  # 对话历史
    query: str                                             # 当前用户问题
    intent: str                                            # 意图: rag_query/chat
    retrieved_docs: List[str]                              # 检索到的文档
    need_rag: bool                                          # 是否需要RAG
    confidence: float                                       # 意图置信度
    steps: List[str]                                        # 执行步骤记录