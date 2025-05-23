import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START

from source_code.agent.llm import qwen_llm_32b

sqlite_client = sqlite3.connect("/source_code/agent/database/db.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_client)


def call_model(state: MessagesState):
    response = qwen_llm_32b.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=memory)
if __name__ == '__main__':

    config = {"configurable": {"thread_id": "1"}}

    # input_message = {"role": "user", "content": "hi! I'm bob"}
    # for chunk in graph.stream({"messages": [input_message]}, config=config,stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()

    input_message = {"role": "user", "content": "what's my name?"}
    for chunk in graph.stream({"messages": [input_message]}, config=config,stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    state = graph
    print(state)
