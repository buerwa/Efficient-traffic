"""
配置文件，包含应用程序配置
"""

import os
from pathlib import Path
from typing import Optional

# 注释掉dotenv依赖
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     print("警告: python-dotenv 未安装，使用默认配置")

# 基本目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR.parent

# API配置
API_V1_STR: str = "/api/v1"

# 安全配置
SECRET_KEY: str = (
    "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # 请在生产环境中更改
)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

# 数据库配置
DATABASE_URL: str = "sqlite:///./traffic_monitor.db"

# 模型配置
MODELS_DIR = BASE_DIR  # 模型保存在backend目录

EFFICIENTNET_MODEL_PATH = os.path.join(MODELS_DIR, "EfficientNet.pth")
# 确保路径存在
print(f"EfficientNet模型路径: {EFFICIENTNET_MODEL_PATH}")
print(f"EfficientNet模型文件存在: {os.path.exists(EFFICIENTNET_MODEL_PATH)}")

# 修复class_indices.json的路径
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
print(f"class_indices.json路径: {CLASS_INDICES_PATH}")
print(f"class_indices.json文件存在: {os.path.exists(CLASS_INDICES_PATH)}")

# 数据存储配置
# 已移除MongoDB配置，使用SQLite代替

# 图片上传配置
UPLOAD_FOLDER: str = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB

# 缓存设置
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 服务器配置
FASTAPI_PORT: int = 8000
DEBUG: bool = True

# 文件上传配置
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = os.path.join(Path(__file__).parent.parent.parent, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 认证配置
SECRET_KEY = "your-secret-key-here"  # 请在生产环境中更改
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 模型配置
MODEL_PATH = os.path.join(MODELS_DIR, "EfficientNet.pth")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
