import os

import fitz
from langchain_community.vectorstores import FAISS
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

from source_code.agent.llm import bge_embeddings
from source_code.api.config import pdf_image_dir, vector_dir

OLLAMA_URL = os.environ.get("OLLAMA_URL")


def transform_pdf_to_txt(pdf_path: str) -> list[dict]:
    os.makedirs(pdf_image_dir, exist_ok=True)
    image_writer = FileBasedDataWriter(pdf_image_dir)
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_path)  # read the pdf content
    ds = PymuDocDataset(pdf_bytes)
    doc_list = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).get_content_list(pdf_image_dir)
    file_name = os.path.basename(pdf_path)
    for doc in doc_list:
        doc.update({"source": file_name})
    return doc_list


def save_embed_text(texts: list[dict], index_name: str) -> None:
    os.makedirs(vector_dir, exist_ok=True)
    documents = []
    metadatas = []
    for i, doc in enumerate(texts):
        documents.append(doc.get("text", ""))
        metadatas.append({"page_idx": doc.get("page_idx", None), "source": doc.get("source", None)})
    db = FAISS.from_texts(documents, bge_embeddings, metadatas=metadatas)
    db.save_local(vector_dir, index_name)


def add_embed_text(texts: list[dict], index_name: str) -> None:
    os.makedirs(vector_dir, exist_ok=True)
    db = FAISS.load_local(vector_dir, bge_embeddings, index_name, allow_dangerous_deserialization=True)
    documents = []
    metadatas = []
    for i, doc in enumerate(texts):
        documents.append(doc.get("text", ""))
        metadatas.append({"page_idx": doc.get("page_idx", None), "source": doc.get("source", None)})
    db.add_texts(texts=documents, metadatas=metadatas)
    db.save_local(vector_dir, index_name)


def convert_pdf_to_images_fitz(pdf_path, output_folder):
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)

    # 遍历每一页
    for page_number in range(pdf_document.page_count):
        # 获取页面
        page = pdf_document[page_number]
        # 将页面转换为图片
        pix = page.get_pixmap()
        # 保存图片
        save_path = os.path.join(output_folder, f"{page_number + 1}.png")
        pix.save(save_path)

    # 关闭PDF文档
    pdf_document.close()
