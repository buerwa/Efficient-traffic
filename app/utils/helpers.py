"""
辅助函数模块
提供各种辅助功能
"""

import os
from typing import Set, Dict, Any, List, Optional
import uuid
import logging
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app.core.config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    检查文件扩展名是否允许

    Args:
        filename: 文件名

    Returns:
        bool: 是否允许
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file_data: bytes, filename: Optional[str] = None) -> str:
    """
    保存上传的文件

    Args:
        file_data: 文件二进制数据
        filename: 可选的文件名

    Returns:
        str: 保存的文件路径
    """
    if filename is None or not allowed_file(filename):
        # 使用UUID作为文件名，扩展名为.jpg
        filename = f"{uuid.uuid4()}.jpg"

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(file_data)
        return file_path
    except Exception as e:
        logger.error(f"保存文件时出错: {e}")
        return ""


def generate_heatmap(image_data: bytes, prediction: Dict[str, Any]) -> str:
    """
    根据预测结果生成热图

    Args:
        image_data: 图像二进制数据
        prediction: 预测结果

    Returns:
        str: 热图的Base64编码字符串
    """
    try:
        # 打开图像
        img = Image.open(io.BytesIO(image_data))

        # 绘制预测结果
        plt.figure(figsize=(12, 6))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img))
        plt.title("原始图像")
        plt.axis("off")

        # 显示预测概率条形图
        plt.subplot(1, 2, 2)

        # 提取并排序概率
        probs = prediction.get("all_probabilities", {})
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

        # 绘制条形图
        plt.bar(range(len(sorted_probs)), list(sorted_probs.values()), align="center")
        plt.xticks(range(len(sorted_probs)), list(sorted_probs.keys()), rotation=45)
        plt.title("预测概率")
        plt.ylabel("概率")
        plt.ylim(0, 1)

        # 添加值标签
        for i, v in enumerate(sorted_probs.values()):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

        plt.tight_layout()

        # 将图像转换为Base64字符串
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        # 关闭图像
        plt.close()

        return img_str

    except Exception as e:
        logger.error(f"生成热图时出错: {e}")
        return ""


def compare_predictions(
    resnet_pred: Dict[str, Any], efficientnet_pred: Dict[str, Any]
) -> Dict[str, Any]:
    """
    比较两个模型的预测结果并生成报告

    Args:
        resnet_pred: ResNet模型的预测结果
        efficientnet_pred: EfficientNet模型的预测结果

    Returns:
        Dict: 包含比较结果的字典
    """
    try:
        # 获取预测结果
        resnet_class = resnet_pred.get("class_name", "unknown")
        efficientnet_class = efficientnet_pred.get("class_name", "unknown")

        # 检查两个模型是否达成一致
        agreement = resnet_class == efficientnet_class

        # 获取置信度
        resnet_conf = resnet_pred.get("confidence", 0)
        efficientnet_conf = efficientnet_pred.get("confidence", 0)

        # 确定推荐的模型
        if agreement:
            if resnet_conf > efficientnet_conf:
                recommendation = "ResNet模型预测置信度更高，推荐采用其结果。"
            else:
                recommendation = "EfficientNet模型预测置信度更高，推荐采用其结果。"
        else:
            # 不一致，选择置信度更高的模型
            if resnet_conf > efficientnet_conf:
                recommendation = f"模型预测结果不一致，但ResNet模型置信度更高({resnet_conf:.2f})，推荐采用其结果。"
            else:
                recommendation = f"模型预测结果不一致，但EfficientNet模型置信度更高({efficientnet_conf:.2f})，推荐采用其结果。"

        return {
            "agreement": agreement,
            "recommendation": recommendation,
            "resnet_prediction": resnet_pred,
            "efficientnet_prediction": efficientnet_pred,
        }

    except Exception as e:
        logger.error(f"比较预测结果时出错: {e}")
        return {"error": str(e), "agreement": False, "recommendation": "比较过程中出错"}
