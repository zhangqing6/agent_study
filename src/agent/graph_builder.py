"""
LangGraph 图构建器
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from .state import AgentState
from .nodes.intent import IntentNode
from .nodes.retrieval import RetrievalNode
from .nodes.generation import GenerationNode
import logging

logger = logging.getLogger(__name__)
class AgentGraphBuilder:
    """LangGraph 图构建器"""

    def __init__(self, config: dict):
        self.config = config
        self.graph = StateGraph(AgentState)

        # 确保配置中包含正确的 Ollama 地址
        ollama_base_url = "http://localhost:11435/v1"
        self.config["ollama_base_url"] = ollama_base_url
        self.config["llm_model"] = "qwen2.5:7b"

        logger.info("初始化 Agent 图构建器")
        logger.info(f"Ollama 地址: {ollama_base_url}")
        logger.info(f"LLM 模型: qwen2.5:7b")

        # 初始化节点
        self.nodes = {
            "intent": IntentNode(self.config),
            "retrieval": RetrievalNode(self.config),
            "generation": GenerationNode(self.config)
        }
        logger.info(f"已创建 {len(self.nodes)} 个节点: {list(self.nodes.keys())}")

    def _add_nodes(self):
        """添加节点"""
        for name, node in self.nodes.items():
            self.graph.add_node(name, node)

    def _add_edges(self):
        """添加边和条件边"""
        # 入口点
        self.graph.set_entry_point("intent")

        # 意图识别后的条件路由
        self.graph.add_conditional_edges(
            "intent",
            self._route_from_intent,
            {
                "retrieval": "retrieval",
                "generation": "generation"
            }
        )

        # 检索后到生成
        self.graph.add_edge("retrieval", "generation")
        self.graph.add_edge("generation", END)

    def _route_from_intent(self, state: AgentState) -> str:
        """意图路由逻辑"""
        if state["intent"] == "rag_query" and state["confidence"] > 0.6:
            return "retrieval"
        return "generation"

    def build(self):
        """构建并编译图"""
        self._add_nodes()
        self._add_edges()

        # 添加检查点实现记忆
        checkpointer = InMemorySaver()
        return self.graph.compile(checkpointer=checkpointer)