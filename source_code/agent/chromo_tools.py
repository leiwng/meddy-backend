import base64
import gzip
import os
import random
import uuid
from typing import List

import cv2
import numpy as np
import requests
from langchain_core.tools import tool

from source_code.api.config import static_dir

rm_bg_server_url = "https://workstation.kemoshen.com/picture/api/picture/rm_bg"
seg_server_url = "https://workstation.kemoshen.com/segmentation/api/ai_seg/seg"
rec_server_url = "https://workstation.kemoshen.com/recognize/api/ai_rec/rec"


@tool
def remove_background_of_original_image(image_path: str) -> str:
    """此函数用于去除染色体图像背景"""
    with open(image_path, 'rb') as f:
        image = f.read()
        response = requests.post(rm_bg_server_url, files={'picture': image})
        if response.status_code == 200:
            json_response = response.json()
            mid_img_data = json_response['result_image']
            # 解码Base64编码的图像数据
            mid_img_data = base64.b64decode(mid_img_data)
            # 将解码后的图像数据转换为 NumPy 数组
            mid_img_array = np.frombuffer(mid_img_data, np.uint8)
            # 将 NumPy 数组解压缩为图像
            mid_img = cv2.imdecode(mid_img_array, cv2.IMREAD_COLOR)
            img_save_path = os.path.join(static_dir, f"mid_img_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(img_save_path, mid_img)
            return img_save_path
        else:
            return "原图去除背景失败"


def find_contour(cv_image, mask_image):
    """
    从掩膜图像中找到轮廓。

    参数:
    - mask_image: 掩膜图像。

    返回:
    - max_contour: 最大的轮廓。

    异常:
    - 如果没有找到轮廓，则返回None。
    """
    mask_image_gray = np.clip(mask_image * 255, 0, 255).astype(np.uint8)
    mid_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    mid_image_gray[mid_image_gray == 0] = 1
    mix_image = cv2.bitwise_and(mask_image_gray, mid_image_gray)
    mid_image_gray[mix_image == 0] = 255
    ret, binary = cv2.threshold(mid_image_gray, 250, 255, cv2.THRESH_BINARY_INV)  # 对灰度图像进行二值化处理
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找轮廓

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    return max_contour


def mask_image_by_contour(cv_img, contour):
    mask = np.zeros(cv_img.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    return cv2.bitwise_and(cv_img, mask)


def draw_one_chromosome_on_224x224(cv_img, contour):
    image_copy = np.copy(cv_img)
    image_copy[image_copy[:, :, 0] == 0] = 1  # 防止某些中期图染色体中有值为0的像素，导致后续操作将染色体这部分像素设置为白色
    new_canvas = mask_image_by_contour(image_copy, contour)

    black_pixels = np.where(
        (new_canvas[:, :, 0] == 0) & (new_canvas[:, :, 1] == 0) & (new_canvas[:, :, 2] == 0)
    )
    new_canvas[black_pixels] = [255, 255, 255]  # 将黑色背景设置为白色

    min_area_rect = cv2.minAreaRect(contour)
    center, size, angle = min_area_rect[0], min_area_rect[1], min_area_rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    if size[0] > size[1]:
        angle = -(90 - angle)

    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = new_canvas[y:y + h, x:x + w]
    cropped_image = cv2.copyMakeBorder(cropped_image, 200, 200, 200, 200, cv2.BORDER_CONSTANT,
                                       value=[255, 255, 255])
    height, width = cropped_image.shape[:2]
    cropped_image_center = (width / 2, height / 2)
    m = cv2.getRotationMatrix2D(cropped_image_center, angle, 1)
    img_rot = cv2.warpAffine(cropped_image, m, (width, height), borderValue=[255, 255, 255])
    gray = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    ret, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)  # 对灰度图像进行二值化处理
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找轮廓
    x, y, w, h = cv2.boundingRect(contours[0])  # 获取轮廓的外接矩形
    img_crop = img_rot[y:y + h, x:x + w]

    dst_high = 224
    dst_wide = 224
    high, wide = img_crop.shape[:2]
    top = int((dst_high - high) / 2)
    down = int((dst_high - high + 1) / 2)
    left = int((dst_wide - wide) / 2)
    right = int((dst_wide - wide + 1) / 2)

    top = 0 if top < 0 else top
    down = 0 if down < 0 else down
    left = 0 if left < 0 else left
    right = 0 if right < 0 else right

    value = [255, 255, 255]
    border_type = cv2.BORDER_CONSTANT
    dst_image = cv2.copyMakeBorder(img_crop, top, down, left, right, border_type, None, value)

    return dst_image, img_crop


def contours_to_224_img(cv_img, contour_list):
    img_bytes_list = []
    cv_224_img_list = []
    for contour in contour_list:
        image, img_crop = draw_one_chromosome_on_224x224(cv_img, contour)
        img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        img_bytes_list.append(img_bytes)
        cv_224_img_list.append(image)
    return cv_224_img_list


@tool(response_format="content_and_artifact")
def segment_mid_image(image_path: str) -> str:
    """此函数用于分割染色体图像"""
    contour_list = []
    cv_img = cv2.imread(image_path)
    img_h, img_w = cv_img.shape[:2]
    with open(image_path, 'rb') as f:
        image = f.read()
        response = requests.post(seg_server_url, files={'picture': image})
        if response.status_code == 200:
            response_json = response.json()
            compressed_mask_list = response_json.get("masks", [])
            for mask in compressed_mask_list:
                try:
                    compressed_array = base64.b64decode(mask)
                    uncompressed_array = gzip.decompress(compressed_array)
                    mask_image = np.frombuffer(uncompressed_array, dtype=np.uint8).reshape((img_h, img_w))
                    contour = find_contour(cv_img, mask_image)
                    contour_list.append(contour)
                except (base64.binascii.Error, gzip.BadGzipFile) as e:
                    raise RuntimeError(f"Failed to decode or decompress mask: {e}")
            cv_img_copy = np.copy(cv_img)
            for contour in contour_list:
                if contour is not None:
                    cv2.drawContours(cv_img_copy, [contour], -1, (0, 255, 0), 1)
            img_save_path = os.path.join(static_dir, f"seg_mid_img_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(img_save_path, cv_img_copy)
            cv_224_img_list = contours_to_224_img(cv_img, contour_list)
            cv_224_img_path_list = []
            for i, cv_224_img in enumerate(cv_224_img_list):
                random_str = ''.join(random.choices('0123456789', k=6))
                seg_img_save_path = os.path.join(static_dir, f"seg_224_{random_str}.jpg")
                cv_224_img_path_list.append(seg_img_save_path)
                cv2.imwrite(seg_img_save_path, cv_224_img)
            return img_save_path, cv_224_img_path_list
        else:
            return "染色体图像分割失败", []


@tool(response_format="content_and_artifact")
def recognize_image(image_list: List[str]):
    """
    使用此函数识别染色体图像
    """
    img_bytes_list = []
    for image_path in image_list:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            img_bytes_list.append(image_bytes)
    files = [("pictures", img_bytes) for img_bytes in img_bytes_list]
    response = requests.post(rec_server_url, files=files, timeout=10)
    if response.status_code == 200:
        response_json = response.json()
        return f"类别: {response_json.get('cls_result', [])} 极性: {response_json.get('polarity', [])}", response_json
    else:
        return "染色体识别失败", {}
