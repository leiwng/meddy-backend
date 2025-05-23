import os

# PATH
# 工程根目录
base_path = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 静态文件路径
static_dir = os.path.join(base_path, 'static')

# 文件路径
file_dir = os.path.join(static_dir, 'file')

# pdf文件路径
pdf_dir = os.path.join(file_dir, 'pdf')

# pdf文件中图像
pdf_image_dir = os.path.join(file_dir, 'pdf_image')

# 向量文件路径
vector_dir = os.path.join(file_dir, 'vector')