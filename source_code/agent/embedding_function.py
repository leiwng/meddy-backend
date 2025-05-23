from source_code.agent.llm import bge_embeddings


async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function that must:
    1. Be async
    2. Accept a list of strings
    3. Return a list of float arrays (embeddings)
    """
    response = await bge_embeddings.aembed_documents(texts)
    return response
