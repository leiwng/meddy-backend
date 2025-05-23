import os
from copy import deepcopy
from typing import List, Literal

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from source_code.agent.chromo_tools import remove_background_of_original_image, segment_mid_image, recognize_image
from source_code.agent.llm import qwen_plus
from source_code.agent.state import State

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "human_in_loop_graph"
os.environ["LANGSMITH_API_KEY"] = "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"

llm_model = qwen_plus


class Plan(BaseModel):
    """做出的计划以及输入的图像数据"""

    steps: List[str] = Field(
        description="需要遵循的不同步骤，应该按顺序排列。"
    )
    input_image: List[str] = Field(
        description="用户的输入图像路径"
    )
    image_type: Literal["microscope_image", "mid_image", "chromo_image"] = Field(
        description="用户的输入图像类型,显微镜图像是microscope_image类型,中期图为mid_image类型,单根染色体图像是chromo_image类型"
    )


class AskHuman(BaseModel):
    """向用户询问问题"""

    question: str


class LatestPlan(BaseModel):
    """
    最首要的任务
    """
    latest_plan: str = Field(
        description="接下来的首要任务"
    )


tools = [Plan, AskHuman]

planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一个专业的染色体图像处理任务规划专家。你的职责是：
            1. 任务分析：
               - 准确理解用户需求（是否需要完整流程或单步处理）
               - 确认输入图像类型和路径
               - 验证图像类型是否满足任务要求
            
            2. 信息收集：
               - 使用AskHuman工具收集缺失信息
               - 图像路径相关问题需使用Markdown格式
            
            3. 计划制定：
               - 使用Plan工具制定执行计划
               - 确保计划步骤清晰且按顺序排列
        
        处理流程参考：
        [显微镜图像] -> 1.去背景 -> [中期图] -> 2.图像实例分割 -> [单根染色体图] -> 3.图像分类识别 -> [识别结果]
        
        注意事项：
        - 不要擅自修改图像路径格式
        - 输入图像可能是任意阶段的图像（显微镜图像/中期图/单根染色体图）
        - 需要根据用户需求和图像类型决定起始步骤
        - 确保每个步骤的输入输出匹配
        """
    ),
    ("placeholder", "{messages}")
])

qwen_llm_with_tools = llm_model.bind_tools(tools=tools)


async def planer_node(state: State) -> Command[Literal["ask_human", "plan_node", "__end__"]]:
    straightforward_messages = []
    for message in state["messages"]:
        copy_message = deepcopy(message)
        if isinstance(copy_message.content, list) and (copy_message.type == "human" or copy_message.type == "ai"):
            copy_message.content = f"{message.content[0]['text']} 图像路径：" + "\n" + "\n".join(
                [f"- {image_content['image_url']}" for image_content in message.content[1:]])
        else:
            pass
        straightforward_messages.append(copy_message)
    message = await qwen_llm_with_tools.ainvoke(
        planner_prompt.invoke({"messages": straightforward_messages})
    )
    if message.tool_calls:
        if message.tool_calls[0]["name"] == "AskHuman":
            return Command(
                goto="ask_human",
                update={"messages": [message]},
            )
        elif message.tool_calls[0]["name"] == "Plan":
            return Command(
                goto="plan_node",
                update={"messages": [message]},
            )
    return Command(
        goto=END,
        update={"messages": [message]}
    )


def ask_human_node(state: State) -> Command[Literal["planer"]]:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    tool_name = state["messages"][-1].tool_calls[0]["name"]
    ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
    answer = interrupt(ask.question)
    return Command(
        goto="planer",
        update={
            "messages": [
                ToolMessage(content=answer, tool_call_id=tool_call_id, name=tool_name),
                AIMessage(content=ask.question),
                HumanMessage(content=answer)
            ]
        },
    )


def plan_node(state: State) -> Command[Literal["replanner"]]:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    tool_name = state["messages"][-1].tool_calls[0]["name"]
    plan_obj = Plan.model_validate(state["messages"][-1].tool_calls[0]["args"])
    plan = plan_obj.steps
    image_type = plan_obj.image_type
    image = plan_obj.input_image

    return Command(
        goto="replanner",
        update={
            "plan": plan,
            image_type: image,
            "messages": [
                ToolMessage(content="计划：" + "\n" + "\n".join(plan), tool_call_id=tool_call_id, name=tool_name)
            ]
        }
    )


replanner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一个任务进度管理专家。你的职责是：
        1. 分析任务状态：
           - 对比原计划和已完成步骤
           - 识别下一个关键任务
        
        2. 任务优先级：
           - 严格按照计划顺序执行
           - 确保任务的连续性和依赖关系
        
        3. 输出要求：
           - 只输出一个最优先的任务
           - 使用明确且可执行的描述
        """
    ),
    ("placeholder", "{messages}")
])

replanner_llm_with_structured_out = llm_model.with_structured_output(LatestPlan, method="function_calling")


async def replanner_node(state: State) -> Command[Literal["check_end_node"]]:
    plans = state["plan"]
    past_steps = state["past_steps"]
    result = await replanner_llm_with_structured_out.ainvoke(
        replanner_prompt.invoke(
            {"messages": [HumanMessage(
                content=f"计划是这样的：{plans}\n已经完成了以下步骤：{past_steps if past_steps else '无'}")]}
        )
    )
    return Command(
        goto="check_end_node",
        update={
            "latest_plan": result.latest_plan if result else ""
        }
    )


check_end_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一个任务完成度验证专家。你的职责是：
        1. 完整性检查：
           - 对比原计划和已完成步骤
           - 验证每个步骤的完成状态
        
        2. 质量验证：
           - 确保所有必要步骤都已执行
           - 验证步骤间的连续性
        
        3. 结果判定：
           - 输出布尔值表示任务是否完成
           - true：所有计划步骤已完成
           - false：还有未完成的步骤
        """
    ),
    ("placeholder", "{messages}")
])


class CheckResponse(BaseModel):
    """使用这个工具来响应用户"""

    is_complete: bool = Field(False, description="是否已完成所有任务")


async def check_end_node(state: State) -> Command[Literal["execute", "responser"]]:
    plans = state["plan"]
    past_steps = state["past_steps"]
    result = await llm_model.with_structured_output(CheckResponse, method="function_calling").ainvoke(
        check_end_prompt.invoke(
            {
                "messages": [
                    HumanMessage(content=f"原有的计划: {plans}"),
                    HumanMessage(content=f"已经完成的工作: {past_steps if past_steps else '无'}"),
                ],
            }
        )
    )
    if result.is_complete:
        return Command(
            goto="responser",
        )
    else:
        return Command(
            goto="execute",
        )


execute_tools = [
    remove_background_of_original_image,
    segment_mid_image,
    recognize_image,
]
tool_node = ToolNode(tools=execute_tools)

execute_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一个染色体图像处理执行专家。你的职责是：
        1. 工具选择：
            - 去背景任务：使用remove_background_of_original_image
            - 分割任务：使用segment_mid_image
            - 识别任务：使用recognize_image
        
        2. 输入验证：
            - 去背景：需要显微镜图像路径
            - 分割：需要中期图路径
            - 识别：需要单根染色体图像路径
        
        3. 执行规范：
            - 只需执行用户指定的单个任务
            - 严格按照任务目标调用对应工具
            - 严格按照工具要求提供输入
            - 确保输入图像类型匹配
        """
    ),
    ("placeholder", "{messages}")
])


async def execute_node(state: State) -> Command[Literal["tool_node"]]:
    task = state["latest_plan"]
    microscope_image_list = state.get('microscope_image', ["无"])
    mid_image_list = state.get('mid_image', ['无'])
    chromo_image_list = state.get('chromo_image', ['无'])

    task_formatted = f"请完成任务:{task}"
    info_formatted = "以下是已知数据：\n" + "1.显微镜图像：\n" + "\n".join(
        [f"- {image}" for image in microscope_image_list]) + "\n" + "2.去背景后产生的中期图：\n" + "\n".join(
        [f"- {image}" for image in mid_image_list]) + "\n" + "3.分割后产生的单根染色体图像：\n" + "\n".join(
        [f"- {image}" for image in chromo_image_list])
    execute_result = await llm_model.bind_tools(tools=execute_tools).ainvoke(
        execute_prompt.invoke(
            {"messages": [HumanMessage(content=task_formatted), HumanMessage(content=info_formatted)]}))
    return Command(
        goto="tool_node",
        update={"messages": [execute_result]}
    )


def tools_result_node(state: State) -> Command[Literal["replanner"]]:
    tool_message = state["messages"][-1]
    if tool_message.name == "remove_background_of_original_image":
        task_result = tool_message.content
        return Command(
            goto="replanner",
            update={
                "past_steps": [("去背景任务执行完成")],
                "mid_image": [task_result]
            }
        )
    elif tool_message.name == "segment_mid_image":
        task_result = tool_message.content
        chromo_image_list = tool_message.artifact
        return Command(
            goto="replanner",
            update={
                "past_steps": [("分割任务执行完成")],
                "chromo_image": chromo_image_list
            }
        )
    elif tool_message.name == "recognize_image":
        task_result = tool_message.content
        return Command(
            goto="replanner",
            update={
                "past_steps": [("识别任务执行完成")],
                "class_info": task_result
            }
        )
    else:
        pass


class Response(BaseModel):
    """Response to user."""
    response: str = Field("", description="总结工作的成果")
    file_path: List[str] = Field([], description="文件路径列表")


response_node_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个专业的染色体图像处理结果分析专家。你的职责是：

            1. 结果分析：
               - 准确理解用户的最终目标
               - 分析处理过程中产生的所有数据
               - 区分任务结果(response)和图像路径(file_path)
            
            2. 结果分类：
               - 去背景任务：
                 * response：说明去背景是否成功
                 * file_path：返回中期图路径
               
               - 分割任务：
                 * response：说明分割出的染色体数量
                 * file_path：返回所有分割后的单根染色体图像路径
               
               - 识别任务：
                 * response：详细说明每个染色体的分类结果
                 * file_path：返回所有分割后的单根染色体图像路径
            
            3. 输出规范：
               - response字段：
                 * 使用清晰简洁的语言描述结果
                 * 针对不同任务类型提供相应的详细信息
                 * 确保信息的完整性和准确性
               
               - file_path字段：
                 * 确保路径列表完整且有效
                 * 确保路径与任务类型对应
            
            注意事项：
            - 始终保持专业性和客观性
            - 确保输出格式符合Response模型的要求
            """
        ),
        ("placeholder", "{messages}"),
    ]
)


async def response_node(state: State) -> Command[Literal["__end__"]]:
    target = state["plan"][-1]
    mid_image_list = state.get('mid_image', ['无'])
    chromo_image_list = state.get('chromo_image', ['无'])
    class_info = state.get('class_info', '无')
    result = await llm_model.with_structured_output(Response, method="function_calling").ainvoke(
        response_node_prompt.invoke(
            {
                "messages": [
                    HumanMessage(content=f"用户的目标: {target}"),
                    HumanMessage(
                        content="以下是处理过程中产生的数据：\n" + "\n" + "去背景后产生的中期图：\n" + "\n".join(
                            [f"- {image}" for image in
                             mid_image_list]) + "\n" + "分割后产生的单根染色体图像：\n" + "\n".join(
                            [f"- {image}" for image in chromo_image_list]) + "\n" + "染色体识别结果：\n" + class_info),
                ],
            }
        )
    )
    response_content = [{"type": "text", "text": result.response}]
    response_image = [{"type": "image_url", "image_url": file_path} for file_path in result.file_path]
    response_content.extend(response_image)
    return Command(
        goto=END,
        update={
            "messages": AIMessage(content=response_content)
        }
    )


workflow = StateGraph(State)

workflow.add_node("planer", planer_node)
workflow.add_node("plan_node", plan_node)
workflow.add_node("ask_human", ask_human_node)
workflow.add_node("replanner", replanner_node)
workflow.add_node("check_end_node", check_end_node)
workflow.add_node("execute", execute_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("tools_result_node", tools_result_node)
workflow.add_node("responser", response_node)

workflow.add_edge("tool_node", "tools_result_node")

workflow.add_edge(START, "planer")

plan_graph = workflow.compile()
