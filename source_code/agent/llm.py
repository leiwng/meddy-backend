import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

os.environ["OPENAI_API_KEY"] = "sk-b4c2b8dca77e454b99399a9ffcd1f9bc"
OLLAMA_URL = os.environ.get("OLLAMA_URL")
VL_MODEL_PATH = os.environ.get("VL_MODEL_PATH")

deepseek_llm_32b = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0.6,
    base_url=OLLAMA_URL,
    # other params...
)

qwen_plus = ChatOpenAI(
    model="qwen-max-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

qwen_llm_32b = ChatOllama(
    model="qwen2.5:32b",
    temperature=0,
    base_url=OLLAMA_URL,
)

bge_embeddings = OllamaEmbeddings(
    model="quentinz/bge-large-zh-v1.5:latest",
    base_url=OLLAMA_URL
)

vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VL_MODEL_PATH, local_files_only=True,
                                                              trust_remote_code=True, torch_dtype="auto",
                                                              device_map="cuda:1")

vl_processor = AutoProcessor.from_pretrained(VL_MODEL_PATH, local_files_only=True, trust_remote_code=True)

model_map = {
    "deepseek-r1:32b": deepseek_llm_32b,
    "qwen2.5:32b": qwen_llm_32b,
    "qwen-plus": qwen_plus
}
