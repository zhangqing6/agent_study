from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class GenerationNode:
    """答案生成节点：根据意图和检索结果生成回答"""

    def __init__(self, config: dict):
        self.llm = ChatOpenAI(
            model=config.get("llm_model", "qwen2.5:7b"),
            temperature=config.get("temperature", 0.7),
            base_url=config.get("ollama_base_url", "http://localhost:11434/v1"),
            api_key="ollama"
        )

    def __call__(self, state: AgentState) -> dict:
        query = state["query"]
        intent = state["intent"]
        docs = state.get("retrieved_docs", [])

        if intent == "rag_query" and docs:
            # RAG 生成
            context = "\n\n".join(docs)
            system_prompt = """你是一个知识助手。请基于提供的知识库内容回答问题。
如果知识库中没有相关信息，请诚实地说不知道，不要编造。"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"知识库内容：\n{context}\n\n问题：{query}")
            ]
        else:
            # 普通对话
            messages = [HumanMessage(content=query)]

        response = self.llm.invoke(messages)

        return {
            "messages": [AIMessage(content=response.content)],
            "steps": ["response_generation"]
        }