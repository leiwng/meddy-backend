import os
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.types import Command

from source_code.agent.configuration import Configuration
from source_code.agent.llm import bge_embeddings, vl_model, vl_processor
from source_code.agent.state import RAGState
from source_code.agent.vision_process import process_vision_info
from source_code.api.config import vector_dir, pdf_dir
from source_code.api.db.mongo import get_db


def retriever_node(state: RAGState, config: RunnableConfig) -> Command[Literal["generation_node"]]:
    rag_expert_name = Configuration.from_runnable_config(config).rag_expert_name
    db = FAISS.load_local(vector_dir, bge_embeddings, index_name=rag_expert_name,
                          allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return Command(
        update={
            "retrieve_text": docs
        }
        ,
        goto="generation_node",
    )


async def generation_node(state: RAGState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    retrieve_text = state["retrieve_text"]
    query = state["messages"][-1].content
    rag_expert_name = Configuration.from_runnable_config(config).rag_expert_name
    mongo_client = await get_db()
    expert_info = await mongo_client.find_one("rag_expert", query={"name": rag_expert_name}, projection={"_id": 0})
    relevant_pdf = expert_info.get("rag_data", [])
    if relevant_pdf:
        relevant_pdf_images = []
        for doc in retrieve_text:
            source = doc.metadata.get("source")
            page_idx = doc.metadata.get("page_idx")
            pdf_image_path = os.path.join(pdf_dir, os.path.splitext(source)[0], f"{page_idx + 1}.png")
            relevant_pdf_images.append(pdf_image_path)
        system_prompt = """
        系统角色：
        你是一个专业的多模态助手，能够同时理解文本和图像信息，并提供准确的回答。你的任务是基于用户的问题和提供的相关图像生成连贯且信息丰富的回答。
        
        任务要求：
        1. 仔细分析提供的图像内容，包括图像中的关键元素、文本、布局和视觉特征
        2. 将图像信息与用户问题进行关联
        3. 生成一个完整的回答，确保：
           - 回答直接针对用户的问题
           - 准确引用图像中的相关信息
           - 保持回答的逻辑性和连贯性
           - 在必要时说明图像中的具体位置或细节
        
        输出格式：
        请提供一个结构化的回答，包含：
        1. 对问题的直接回应
        2. 基于图像的具体论述
        3. 必要的补充说明或建议
        
        注意事项：
        - 如果图像中的信息不足以完全回答问题，请明确指出
        - 保持专业和客观的语气
        - 避免做出无法从图像验证的假设
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ]
        for image in relevant_pdf_images:
            messages[1]["content"].append({"type": "image", "image": image})
        text = vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:1")
        generated_ids = vl_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vl_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return_content = [{"type": "text", "text": output_text}]
        for i, image in enumerate(relevant_pdf_images):
            return_content.append({"type": "image_url", "image_url": image})
        return Command(
            update={
                "messages": AIMessage(content=return_content)
            }
            ,
            goto="__end__",
        )
    else:
        return Command(
            update={
                "messages": AIMessage(content="没有找到相关数据")
            }
            ,
            goto="__end__",
        )


workflow = StateGraph(RAGState)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("generation_node", generation_node)
workflow.add_edge(START, "retriever_node")
rag_workflow = workflow.compile()
