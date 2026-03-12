"""
答案生成节点
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import logging
from ..state import AgentState

logger = logging.getLogger(__name__)


class GenerationNode:
    """答案生成节点：根据意图和检索结果生成回答"""

    def __init__(self, config: dict):
        self.llm = ChatOpenAI(
            model="qwen2.5:7b",
            temperature=config.get("temperature", 0.7),
            base_url="http://localhost:11435/v1",
            api_key="ollama",
            timeout=60
        )
        logger.info(f"生成节点初始化")

    def _format_history(self, messages) -> str:
        """将历史消息格式化为纯文本，让模型能理解对话上下文"""
        if not messages:
            return ""

        history = "以下是我们的对话历史：\n"
        for i, msg in enumerate(messages):
            # 判断消息类型
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    history += f"用户：{msg.content}\n"
                elif msg.type == "ai":
                    history += f"助手：{msg.content}\n"
                else:
                    history += f"系统：{msg.content}\n"
            else:
                # 如果是字符串或其他格式
                history += f"{msg}\n"

        return history

    def __call__(self, state: AgentState) -> dict:
        query = state["query"]
        intent = state["intent"]
        docs = state.get("retrieved_docs", [])

        # 从 state 中获取完整的消息历史 - 这是关键！
        messages_history = state.get("messages", [])

        logger.info(f"生成回答 - 意图: {intent}, 检索文档数: {len(docs)}")
        logger.info(f"历史消息数: {len(messages_history)}")

        # 如果有历史，打印前两条看看
        if messages_history:
            logger.info(f"历史消息示例: {messages_history[0].content[:50]}...")

        # 构建消息列表
        if intent == "rag_query" and docs:
            # RAG 模式：基于知识库回答，不带历史（因为知识库内容会覆盖）
            context = "\n\n".join(docs)
            system_prompt = """你是一个知识助手。请基于提供的知识库内容回答问题。"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"知识库内容：\n{context}\n\n问题：{query}")
            ]
        else:
            # 对话模式：必须带上历史！
            messages = list(messages_history)  # 👈 把历史加进来
            messages.append(HumanMessage(content=query))
            logger.info(f"对话模式，总共 {len(messages)} 条消息")

        try:
            response = self.llm.invoke(messages)

            # 返回时，LangGraph 会自动把这次对话加到 state['messages'] 中
            return {
                "messages": [AIMessage(content=response.content)],
                "steps": ["response_generation"]
            }

        except Exception as e:
            logger.error(f"回答生成失败: {e}")
            return {
                "messages": [AIMessage(content=f"生成回答时出错: {str(e)}")],
                "steps": ["response_generation", "error"]
            }