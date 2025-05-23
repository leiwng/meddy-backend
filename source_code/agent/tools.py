import uuid
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

from source_code.agent.configuration import Configuration

@tool
def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: Optional[uuid.UUID] = None,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore],
):
    """在数据库中插入一段记忆

    如果某个记忆与现有记忆冲突，则只需通过传入memory_id 来更新现有记忆即可 - 不要创建两个相同的记忆。如果用户更正了某个记忆，则更新该记忆。

    Args:
        content: 记忆的主要内容。例如：“用户表示有兴趣学习法语。”
        context: 记忆的上下文信息。例如：“在讨论欧洲的职业选择时提到了这一点。”
        memory_id: 仅在更新现有内存时提供。
    """
    mem_id = memory_id or uuid.uuid4()
    user_id = Configuration.from_runnable_config(config).user_id
    store.put(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"记忆创建成功，id为{mem_id}"