import json
import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from source_code.agent.configuration import Configuration
from source_code.agent.human_in_loop_graph import plan_graph
from source_code.agent.llm import qwen_llm_32b, model_map
from source_code.agent.rag_graph import rag_workflow
from source_code.agent.state import GraphState
from source_code.agent.subgraph_test import subgraph
from source_code.agent.tools import upsert_memory
from source_code.api.web_socket import server

logger = logging.getLogger(__name__)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "therapy_flow"
os.environ["LANGSMITH_API_KEY"] = "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"

llm_model = qwen_llm_32b


@tool
def get_weather(city: Literal["成都", "北京"]):
    """使用这个函数获取天气信息"""
    if city == "成都":
        return "成都天气很好"
    elif city == "北京":
        return "北京天气很好"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]

members = ["tool", "chat", "chromo_tool", "subgraph", "expert", "upsert_memory"]

supervisor_node_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个判断对话意图的专家，你的任务是根据用户对话内容判断用户意图并使用ResponseFormatter返回下一个处理节点。
            
            可用节点及其职责：
            1. 'tool': 处理工具调用请求
               - 天气查询等通用工具调用
               - 需要明确的工具名称或功能描述
            
            2. 'chromo_tool': 染色体图像处理
               - 图像去背景
               - 图像分割
               - 染色体识别和分类
               - 任何与染色体图像分析相关的操作
            
            3. 'upsert_memory': 记忆管理
               - 用户希望系统记住某些信息
               - 用户陈述个人偏好
               - 包含"记住"、"记录"等记忆相关词汇
               - 需要存储的重要用户信息
            
            4. 'chat': 通用对话
               - 不属于以上类别的一般性对话
               - 问候、闲聊等日常交互
               - 专业咨询但不需要工具支持
            
            5. 'subgraph': 子图测试（仅在用户明确要求时使用）
            
            判断规则：
            1. 优先级：专业工具 > 通用工具 > 记忆存储 > 普通对话
            2. 如有多个意图，选择最关键的一个处理
            3. 如果不确定，默认转到 'chat' 节点
            
            注意：禁止直接回复用户消息。请仔细分析用户输入中的关键词和上下文，确保准确路由到合适的处理节点。
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

chat_node_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是由科莫生科技有限公司开发的智能体医小助，一个专注于染色体分析和医疗对话的AI助手。
            
            核心能力：
                1. 染色体专业技能
                   - 染色体图像背景分割
                   - 染色体实例分割
                   - 染色体分类识别
                   - 染色体相关专业知识解答
                
            记忆管理：
               1. 用户记忆：
                   <memories>
                   {memories}
                   </memories>
            
            对话原则：
               1. 专业性
                  - 使用准确的医学术语
                  - 保持专业的表达方式
                  - 在涉及专业问题时提供详细解释
               
               2. 个性化
                  - 结合用户历史记忆进行对话
                  - 注意用户特定需求和偏好
                  - 保持对话的连续性
               
               3. 安全性
                  - 不提供具体医疗建议
                  - 对敏感问题保持谨慎
                  - 明确表示AI身份
            
            回复规范：
               - 保持友好专业的语气
               - 回答要简洁明确
               - 必要时提供知识扩展
               - 遇到不确定内容时明确表示
            
            特殊情况处理：
               - 遇到超出能力范围的问题时，建议咨询专业医生
               - 涉及隐私信息时，提醒用户注意信息安全
               - 技术问题时，提供可行的解决方案
            
            注意：始终以专业、负责任的态度进行对话，确保信息的准确性和可靠性。
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)


class SupervisorNodeOutputFormatter(BaseModel):
    """返回需要前往的节点"""
    next_worker: Literal[*members] = Field(description="需要前往的下一个节点名称")


async def supervisor_node(state: GraphState, config: RunnableConfig) -> Command[Literal[*members, "__end__"]]:
    is_expert_chat_mode = Configuration.from_runnable_config(config).expert_chat_mode
    is_rag_chat_mode = Configuration.from_runnable_config(config).rag_chat_mode
    if is_expert_chat_mode:
        return Command(
            goto="expert",
        )
    elif is_rag_chat_mode:
        return Command(
            goto="rag",
        )
    response = await llm_model.with_structured_output(SupervisorNodeOutputFormatter, method="function_calling").ainvoke(
        supervisor_node_prompt.invoke(
            {"members": members, "messages": [state["messages"][-1]]}))
    if hasattr(response, "next_worker"):
        goto = response.next_worker
    else:
        goto = "chat"

    return Command(
        goto=goto
    )


tool_agent = create_react_agent(
    llm_model, tools=tools, prompt="你是一个带有工具调用的智能体"
)

upsert_memory_agent = create_react_agent(
    llm_model, tools=[upsert_memory], prompt="你是一个带有工具调用的智能体, 请使用upsert_memory记录用户记忆"
)


def tool_node(state: GraphState) -> Command[Literal["__end__"]]:
    last_five_messages = state["messages"][-4:]
    logger.info(last_five_messages)
    result = tool_agent.invoke(last_five_messages)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="tool")
            ]
        },
        goto=END,
    )


async def chat_node(state: GraphState, config: RunnableConfig, *, store: BaseStore) -> Command[Literal["__end__"]]:
    memories = await store.asearch(
        ("memories", Configuration.from_runnable_config(config).user_id),
        query=str([m.content for m in state["messages"][-3:]]),
        limit=10,
    )
    formatted_memories = "\n".join(f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories)
    formatted_memories = formatted_memories if formatted_memories else "暂未记录用户记忆"
    model = model_map[Configuration.from_runnable_config(config).model_name]
    result = await model.ainvoke(
        chat_node_prompt.invoke({"memories": formatted_memories, "messages": state["messages"]}))
    return Command(
        update={
            "messages": [
                AIMessage(content=result.content, name="chat")
            ]
        },
        goto=END,
    )


async def upsert_memory_node(state: GraphState, config: RunnableConfig, *, store: BaseStore) -> Command[
    Literal["__end__"]]:
    result = await upsert_memory_agent.ainvoke({"messages": state["messages"]})
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="upsert_memory")
            ]
        },
        goto=END,
    )


async def chromo_tool_node(state: GraphState) -> Command[Literal["__end__"]]:
    last_five_messages = state["messages"]
    result = await plan_graph.ainvoke(
        {"messages": last_five_messages})
    return Command(
        update={
            "messages": result["messages"]
        },
        goto=END,
    )


def subgraph_node(state: GraphState) -> Command[Literal["__end__"]]:
    result = subgraph.invoke({"messages": state["messages"]})
    return Command(
        update={
            "messages": result["messages"]
        },
        goto=END,
    )


async def expert_node(state: GraphState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    thread_id = Configuration.from_runnable_config(config).thread_id
    await server.broadcast(json.dumps({
        "thread_id": thread_id,
        "content": state["messages"][-1].content,
        "additional_kwargs": state["messages"][-1].additional_kwargs
    }))
    return Command(
        update={
            "messages": state["messages"][-1]
        },
        goto=END
    )


builder = StateGraph(GraphState, config_schema=Configuration)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("tool", tool_agent)
builder.add_node("chat", chat_node)
builder.add_node("upsert_memory", upsert_memory_node)
builder.add_node("chromo_tool", chromo_tool_node)
builder.add_node("subgraph", subgraph_node)
builder.add_node("expert", expert_node)
builder.add_node("rag", rag_workflow)
graph = builder.compile()
