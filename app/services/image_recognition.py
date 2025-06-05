import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


class EfficientNetModel:
    def __init__(self, model_path, class_indices_path):
        # 确定设备，但强制使用CPU避免CUDA错误
        self.device = torch.device("cpu")
        self.model = self._load_model(model_path)
        self.class_indices = self._load_class_indices(class_indices_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # ImageNet标准化
            ]
        )
        logger.info(f"模型加载成功，使用设备: {self.device}")

    def _load_model(self, model_path):
        """加载预训练的EfficientNet模型"""
        try:
            # 创建模型
            model = EfficientNet.from_pretrained("efficientnet-b1", num_classes=4)

            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # 检查checkpoint格式并加载模型权重
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # 如果是包含训练信息的checkpoint
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("从checkpoint中加载模型权重")
            else:
                # 如果是直接的state_dict
                model.load_state_dict(checkpoint)
                logger.info("直接加载模型权重")

            # 将模型移到正确的设备上
            model = model.to(self.device)

            # 设置为评估模式 (关闭Dropout等训练特性)
            model.eval()

            logger.info(f"模型已加载: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _load_class_indices(self, class_indices_path):
        """加载类别索引文件"""
        try:
            with open(class_indices_path, "r", encoding="utf-8") as f:
                class_indices = json.load(f)
                logger.info(f"类别索引已加载: {class_indices}")
                return class_indices
        except Exception as e:
            logger.error(f"加载类别索引失败: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """预处理图像"""
        try:
            # 加载并转换图像
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            # 添加批次维度并移到正确设备
            image = image.unsqueeze(0).to(self.device)
            return image
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise

    def predict(self, image_path):
        """对图像进行预测"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            # 预处理图像
            image = self.preprocess_image(image_path)
            logger.info(f"图像已预处理: {image_path}")

            # 进行预测
            with torch.no_grad():
                # 获取模型输出并计算概率
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # 获取最高概率的预测结果
                confidence, predicted_idx = torch.max(probabilities, 1)

                # 转换为Python标量
                predicted_idx = predicted_idx.item()
                confidence_value = confidence.item()

                # 获取类别名称
                prediction = self.class_indices[str(predicted_idx)]

                logger.info(f"预测结果: {prediction}, 可信度: {confidence_value:.4f}")

                # 添加所有类别的概率分布
                all_probs = {}
                for idx, prob in enumerate(probabilities[0].cpu().numpy()):
                    class_name = self.class_indices[str(idx)]
                    all_probs[class_name] = float(prob)

                return {
                    "prediction": prediction,
                    "confidence": confidence_value,
                    "all_probabilities": all_probs,
                }
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
