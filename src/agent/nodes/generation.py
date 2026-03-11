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

        # 获取历史消息
        messages_history = state.get("messages", [])

        logger.info(f"历史消息数: {len(messages_history)}")

        try:
            # 判断是否是记忆相关问题
            memory_keywords = ["我叫什么", "我的名字", "我刚才", "之前", "上一句", "还记得"]
            is_memory_question = any(kw in query for kw in memory_keywords)

            if is_memory_question:
                # 记忆问题：完全忽略知识库，只用历史
                logger.info("检测到记忆问题，忽略知识库，只用历史")

                # 构建历史对话
                history_text = ""
                for msg in messages_history:
                    if hasattr(msg, 'type'):
                        if msg.type == "human":
                            history_text += f"用户：{msg.content}\n"
                        elif msg.type == "ai":
                            history_text += f"助手：{msg.content}\n"

                prompt = f"""以下是我们的对话历史：
    {history_text}

    用户现在问：{query}

    请根据对话历史回答。如果用户在历史中告诉过你名字或其他信息，请直接使用这些信息回答。
    不要编造，如果历史中没有相关信息，就如实说不知道。
    """
                messages = [HumanMessage(content=prompt)]

            elif intent == "rag_query" and docs:
                # RAG模式：但弱化提示词，允许结合自身知识
                context = "\n\n".join(docs)

                # 修改后的提示词 - 优先级：历史 > 知识库 > 模型知识
                system_prompt = """你是一个智能助手。回答问题时请遵循以下优先级：
    1. 如果问题涉及对话历史（如问名字、问之前说过什么），优先使用对话历史回答
    2. 如果提供了知识库内容，可以参考知识库
    3. 最后可以用你自己的知识补充

    注意：不要因为有了知识库就忽略对话历史！
    """
                user_prompt = f"对话历史已提供给你。\n\n知识库内容：\n{context}\n\n用户问题：{query}"

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                logger.info("使用改进的RAG模式（兼顾历史）")

            else:
                # 普通对话：只带历史
                history_text = ""
                for msg in messages_history:
                    if hasattr(msg, 'type'):
                        if msg.type == "human":
                            history_text += f"用户：{msg.content}\n"
                        elif msg.type == "ai":
                            history_text += f"助手：{msg.content}\n"

                prompt = f"""以下是我们的对话历史：
    {history_text}

    用户现在问：{query}

    请根据对话历史回答。"""
                messages = [HumanMessage(content=prompt)]
                logger.info("使用普通对话模式")

            response = self.llm.invoke(messages)

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