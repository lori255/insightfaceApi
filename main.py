import asyncio
import logging
import os
import sys
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import io

import cv2
import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, Header, UploadFile, File, Form
from insightface.app import FaceAnalysis
from insightface.utils import storage

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# 从环境变量中加载配置
api_auth_key = os.getenv("API_AUTH_KEY", "112233")  # 默认值为 "112233"
http_port = int(os.getenv("HTTP_PORT", "8066"))  # 默认端口 8066
detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")  # 默认后端 insightface
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")  # 默认模型 buffalo_l
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))  # 默认阈值 0.65
max_file_size = int(os.getenv("MAX_FILE_SIZE", "10"))  # 最大文件大小，单位 MB
semaphore_limit = int(os.getenv("SEMAPHORE_LIMIT", "10"))  # 最大并发数

# 设置模型存储路径
storage.BASE_REPO_URL = os.getenv(
    "MODEL_STORAGE_URL",
    "https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models"
)

# 初始化人脸分析器
faceAnalysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                            allowed_modules=['detection', 'recognition'], name=recognition_model)
faceAnalysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))

# 检查活动状态
inactive_task = None
semaphore = asyncio.Semaphore(semaphore_limit)
executor = ThreadPoolExecutor(max_workers=semaphore_limit)


async def check_inactive():
    """检查服务是否长时间未活动，重启程序"""
    await asyncio.sleep(3600)  # 1小时无活动则重启
    restart_program()


def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


@app.middleware("http")
async def check_activity(request, call_next):
    """活动状态监控中间件"""
    global inactive_task
    if inactive_task:
        inactive_task.cancel()
    inactive_task = asyncio.create_task(check_inactive())
    response = await call_next(request)
    return response


@app.middleware("http")
async def log_requests(request, call_next):
    """日志记录中间件"""
    logging.info(f"收到请求: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"响应状态: {response.status_code}")
    return response


# 验证API密钥
async def verify_header(api_key: str = Header(...)):
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    return api_key


# 图像处理
async def process_image(image_file=None, image_base64=None, image_url=None):
    if image_file:
        return await read_image_file(image_file)
    if image_base64:
        return decode_base64_image(image_base64)
    if image_url:
        return await download_and_process_image(image_url)
    raise HTTPException(status_code=400, detail="请提供图像文件、Base64编码或图像URL")


async def read_image_file(image_file: UploadFile):
    if image_file.size > max_file_size * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件大小超过{max_file_size}MB限制")
    contents = await image_file.read()
    return cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)


def decode_base64_image(image_base64: str):
    # 计算Base64编码大小
    base64_size = (len(image_base64) * 3) / 4 - image_base64.count('=')
    if base64_size > max_file_size * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Base64图片大小超过{max_file_size}MB限制")

    img_data = base64.b64decode(image_base64)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析Base64图片")
    return img


async def download_and_process_image(image_url: str):
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="下载图片失败")

    # 检查内容大小
    content_length = response.headers.get('Content-Length')
    if content_length and int(content_length) > max_file_size * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"图片大小超过{max_file_size}MB限制")

    # 加载图像内容
    image_bytes = io.BytesIO(response.content)
    img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片")
    return img


async def predict(func, img):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, img)


def _represent(img):
    faces = faceAnalysis.get(img)
    results = []
    for face in faces:
        resp_obj = {
            "embedding": face.normed_embedding.astype(float).tolist(),
            "facial_area": {
                "x": int(face.bbox[0]),
                "y": int(face.bbox[1]),
                "w": int(face.bbox[2] - face.bbox[0]),
                "h": int(face.bbox[3] - face.bbox[1])
            },
            "face_confidence": float(face.det_score)
        }
        results.append(resp_obj)
    return results


def calculate_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# API 路由
@app.get("/")
async def top_info():
    return {
        "title": "人脸识别API",
        "link": "https://github.com/deepinsight/insightface",
        "detector_backend": detector_backend,
        "recognition_model": recognition_model
    }


@app.post("/represent")
async def process_image_api(
        image_file: Optional[UploadFile] = File(None),
        image_base64: Optional[str] = Form(None),
        image_url: Optional[str] = Form(None),
        api_key: str = Depends(verify_header)
):
    async with semaphore:
        img = await process_image(image_file, image_base64, image_url)
        embedding_objs = await predict(_represent, img)
        return {"result": embedding_objs}


@app.post("/compare")
async def compare_faces(
        image_file1: Optional[UploadFile] = File(None),
        image_file2: Optional[UploadFile] = File(None),
        image_base64_1: Optional[str] = Form(None),
        image_base64_2: Optional[str] = Form(None),
        image_url1: Optional[str] = Form(None),
        image_url2: Optional[str] = Form(None),
        api_key: str = Depends(verify_header)
):
    async with semaphore:
        img1 = await process_image(image_file1, image_base64_1, image_url1)
        img2 = await process_image(image_file2, image_base64_2, image_url2)

        embeddings1 = await predict(_represent, img1)
        embeddings2 = await predict(_represent, img2)

        if embeddings1 and embeddings2:
            max_similarity = float('-inf')
            for face1 in embeddings1:
                for face2 in embeddings2:
                    similarity = calculate_similarity(face1["embedding"], face2["embedding"])
                    if similarity > max_similarity:
                        max_similarity = similarity

            return {"max_similarity": max_similarity}

        raise HTTPException(status_code=400, detail="无法从一张或两张图像中提取特征。")


# 启动程序
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=http_port)
