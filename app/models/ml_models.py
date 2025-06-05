"""
机器学习模型加载和预测的实现
"""

import os
import json
import torch
from PIL import Image
import io
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, Any, List, Optional, Union
import time
import logging

from app.core.config import (
    RESNET_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH,
    CLASS_INDICES_PATH,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 预处理转换
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class ModelManager:
    """模型管理类，负责加载和使用各种模型"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models = {}
        self.class_indices = self._load_class_indices()
        self._initialized = True

    def _load_class_indices(self) -> Dict[str, str]:
        """加载类别索引"""
        try:
            with open(CLASS_INDICES_PATH, "r") as json_file:
                class_indict = json.load(json_file)
            logger.info("成功加载类别索引")
            return class_indict
        except Exception as e:
            logger.error(f"加载类别索引出错: {e}")
            # 默认类别索引
            return {
                "0": "accident",
                "1": "dense_traffic",
                "2": "fire",
                "3": "sparse_traffic",
            }

    def load_model(self, model_name: str) -> bool:
        """
        加载指定的模型

        Args:
            model_name: 模型名称 ('resnet' 或 'efficientnet')

        Returns:
            bool: 是否成功加载模型
        """
        try:
            if model_name == "resnet":
                from model import ResNet

                model = ResNet(num_classes=len(self.class_indices))
                model_path = RESNET_MODEL_PATH
            elif model_name == "efficientnet":
                from model_eff import EfficientNet

                model = EfficientNet(num_classes=len(self.class_indices))
                model_path = EFFICIENTNET_MODEL_PATH
                logger.info(f"EfficientNet模型路径: {model_path}")
            else:
                logger.error(f"不支持的模型: {model_name}")
                return False

            if os.path.exists(model_path):
                logger.info(f"加载模型文件: {model_path}")

                try:
                    model.load_state_dict(torch.load(model_path))
                    model.eval()  # 设置为评估模式
                    self.models[model_name] = model
                    logger.info(f"{model_name} 模型加载成功")
                    return True
                except Exception as e:
                    logger.error(f"加载模型权重时出错: {e}")
                    return False
            else:
                abs_path = os.path.abspath(model_path)
                logger.error(f"模型文件不存在: {abs_path}")

                # 尝试直接从根目录加载
                if model_name == "efficientnet":
                    root_model_path = os.path.join(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        ),
                        "EfficientNet.pth",
                    )
                    logger.info(f"尝试从根目录加载EfficientNet: {root_model_path}")

                    if os.path.exists(root_model_path):
                        try:
                            model.load_state_dict(torch.load(root_model_path))
                            model.eval()
                            self.models[model_name] = model
                            logger.info(f"{model_name} 从根目录加载成功")
                            return True
                        except Exception as e:
                            logger.error(f"从根目录加载模型权重时出错: {e}")
                            return False

                return False

        except Exception as e:
            logger.error(f"加载 {model_name} 模型时出错: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def get_model(self, model_name: str) -> Any:
        """
        获取指定的模型，如果未加载则尝试加载

        Args:
            model_name: 模型名称

        Returns:
            模型实例或None（如果加载失败）
        """
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                return None

        return self.models.get(model_name)

    def predict(self, image_data: bytes, model_name: str = "resnet") -> Dict[str, Any]:
        """
        使用指定模型对图像进行预测

        Args:
            image_data: 图像的二进制数据
            model_name: 要使用的模型名称

        Returns:
            包含预测结果的字典
        """
        start_time = time.time()

        try:
            # 获取模型
            model = self.get_model(model_name)
            if model is None:
                return {"error": f"无法加载 {model_name} 模型"}

            # 读取和预处理图像
            img = Image.open(io.BytesIO(image_data))
            img_tensor = data_transform(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)  # 添加批次维度

            # 预测
            with torch.no_grad():
                output = model(img_tensor)
                output = torch.squeeze(output)  # 移除批次维度
                predict_probs = torch.softmax(output, dim=0)
                predict_class = torch.argmax(predict_probs).item()

                # 获取所有类别的概率
                all_probs = {
                    self.class_indices[str(i)]: float(predict_probs[i])
                    for i in range(len(predict_probs))
                }

                processing_time = time.time() - start_time

                result = {
                    "class_name": self.class_indices[str(predict_class)],
                    "class_id": predict_class,
                    "confidence": float(predict_probs[predict_class]),
                    "all_probabilities": all_probs,
                    "model_used": model_name,
                    "processing_time_ms": round(processing_time * 1000, 2),
                }

                return result

        except Exception as e:
            logger.error(f"预测过程中出错: {e}")
            return {"error": f"预测过程中出错: {str(e)}"}

    def compare_models(self, image_data: bytes) -> Dict[str, Any]:
        """
        比较不同模型的预测结果

        Args:
            image_data: 图像的二进制数据

        Returns:
            包含不同模型预测结果的字典
        """
        results = {}

        for model_name in ["resnet", "efficientnet"]:
            result = self.predict(image_data, model_name)
            results[model_name] = result

        return results
