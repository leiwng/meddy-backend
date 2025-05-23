from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
from langchain_core.messages import AIMessage

from source_code.agent.state import GraphState


def subgraph_node_1(state: GraphState):
    return {"messages": AIMessage("from subgraph_node_1")}


def subgraph_node_2(state: GraphState):
    # note that this node is using a state key ('bar') that is only available in the subgraph
    # and is sending update on the shared state key ('foo')
    return Command(
        goto=END,
        update={"messages": AIMessage("from subgraph_node_2")},
    )

subgraph_builder = StateGraph(GraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()
