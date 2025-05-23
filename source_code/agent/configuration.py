"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Literal

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # Changeme: Add configurable values here!
    # these values can be pre-set when you
    # create assistants (https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/)
    # and when you invoke the graph
    model_name: Literal["deepseek-r1:32b", "qwen2.5:32b", "qwen-plus"] = "deepseek-r1:32b"
    thread_id: str = ""
    expert_chat_mode: bool = False
    rag_chat_mode: bool = False
    rag_expert_name : str = ""
    user_id: str = ""

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


if __name__ == '__main__':
    config = Configuration.from_runnable_config()
    print(config.model_name)
