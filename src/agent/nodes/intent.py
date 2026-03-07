from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json


class IntentNode:
    """意图识别节点：判断用户问题是否需要知识检索"""

    def __init__(self, config: dict):
        self.llm = ChatOpenAI(
            model=config.get("llm_model", "qwen2.5:7b"),
            temperature=0.1,
            base_url=config.get("ollama_base_url", "http://localhost:11434/v1"),
            api_key="ollama"  # Ollama 不需要真实 API key
        )

    def __call__(self, state: AgentState) -> dict:
        query = state["query"]

        system_prompt = """你是一个意图识别专家。分析用户问题的意图，返回JSON格式。

可选的意图：
- rag_query: 需要查询知识库（技术问题、文档查询、事实性问题）
- chat: 普通对话（问候、闲聊、观点表达）

返回格式：{"intent": "rag_query/chat", "confidence": 0.0-1.0}
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        response = self.llm.invoke(messages)

        try:
            result = json.loads(response.content)
            intent = result["intent"]
            confidence = result["confidence"]
        except:
            # 默认走聊天流程
            intent = "chat"
            confidence = 0.5

        return {
            "intent": intent,
            "confidence": confidence,
            "steps": ["intent_recognition"]
        }