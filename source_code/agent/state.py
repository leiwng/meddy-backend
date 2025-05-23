import operator
from typing import List, Tuple, Annotated

from langgraph.graph import MessagesState
from langchain_core.documents import Document


class State(MessagesState):
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    latest_plan: str
    microscope_image: List[str]
    mid_image: List[str]
    chromo_image: List[str]
    class_info: str
    response: str
    file_path: List[str]

class GraphState(MessagesState):
    next :str

class RAGState(MessagesState):
    retrieve_text: list[Document]