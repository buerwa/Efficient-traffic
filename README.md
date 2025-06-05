# 🚦 交通实况识别系统 - 后端服务

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于深度学习的智能交通实况识别系统后端，使用 **EfficientNet** 模型实现对交通图像的实时分析，能够准确识别四种交通状况：交通事故、交通拥堵、车辆起火和正常行驶。

## ✨ 主要特性

- 🧠 **智能识别**: 采用 EfficientNet 深度学习模型，高精度识别交通状况
- 🚀 **高性能 API**: 基于 FastAPI 框架，提供高并发、低延迟的 RESTful 接口
- 🔐 **安全认证**: JWT 身份验证和权限管理系统
- 📊 **数据统计**: 完整的历史记录查询和统计分析功能
- 📱 **实时处理**: 支持实时摄像头图像和批量图像处理
- 📖 **API 文档**: 自动生成的 Swagger UI 文档

## 🎯 识别类别

系统能够识别以下四种交通状况：

| 类别        | 英文标识         | 描述               |
| ----------- | ---------------- | ------------------ |
| 🚗 交通顺畅 | `sparse_traffic` | 道路畅通，车辆稀少 |
| 🚦 交通拥堵 | `dense_traffic`  | 道路拥挤，车辆密集 |
| 💥 交通事故 | `accident`       | 发生交通事故现场   |
| 🔥 车辆起火 | `fire`           | 车辆起火或其他火灾 |

## 🛠️ 技术栈

- **Web 框架**: FastAPI 0.68+
- **深度学习**: PyTorch 2.0+, EfficientNet
- **数据库**: SQLite
- **身份验证**: JWT (JSON Web Tokens)
- **图像处理**: OpenCV, Pillow
- **数据分析**: NumPy, Pandas, Scikit-learn
- **部署**: Uvicorn ASGI 服务器

## 📋 系统要求

- Python 3.9 或更高版本
- **推荐**: Anaconda（便于环境管理和依赖安装）
- 内存: 至少 4GB RAM
- 存储: 至少 1GB 可用空间
- 操作系统: Windows/Linux/macOS

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/buerwa/Efficient-traffic.git
```

### 2. 创建虚拟环境（推荐使用 Anaconda）

```bash
# 使用 Anaconda/Miniconda（推荐）
conda create -n traffic-recognition python=3.9
conda activate traffic-recognition

# 或者使用 venv
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 如果使用 Anaconda（推荐）
pip install -r requirements.txt

# 或者优先使用 conda 安装主要依赖
conda install pytorch torchvision opencv numpy pandas scikit-learn
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python run.py
```

服务启动后，您可以访问：

- **API 服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc

## 📡 API 接口

### 认证接口

- `POST /api/v1/token` - 用户登录获取访问令牌

### 预测接口

- `POST /api/v1/predict` - 单张图像分析
- `POST /api/v1/predict/batch` - 批量图像分析
- `POST /api/v1/predict/camera` - 实时摄像头分析

### 数据管理

- `GET /api/v1/predictions` - 获取历史预测记录
- `GET /api/v1/predictions/{id}` - 获取特定预测详情
- `DELETE /api/v1/predictions/{id}` - 删除预测记录

### 统计分析

- `GET /api/v1/statistics` - 获取系统统计信息
- `GET /api/v1/statistics/daily` - 获取每日统计
- `GET /api/v1/statistics/trends` - 获取趋势分析

## 💡 使用示例

### Python 客户端示例

```python
import requests

# 1. 登录获取token
login_data = {"username": "admin", "password": "admin123"}
response = requests.post("http://localhost:8000/api/v1/token", data=login_data)
token = response.json()["access_token"]

# 2. 上传图像进行分析
headers = {"Authorization": f"Bearer {token}"}
files = {"file": open("traffic_image.jpg", "rb")}
response = requests.post("http://localhost:8000/api/v1/predict",
                        files=files, headers=headers)
result = response.json()
print(f"识别结果: {result['prediction']}")
print(f"置信度: {result['confidence']:.2%}")
```

### cURL 示例

```bash
# 登录
curl -X POST "http://localhost:8000/api/v1/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=admin123"

# 图像分析
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@traffic_image.jpg"
```

## 🔧 配置说明

### 环境变量

创建 `.env` 文件来配置系统参数：

```env
# 服务器配置
FASTAPI_PORT=8000
DEBUG=false

# 安全配置
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 数据库配置
DATABASE_URL=sqlite:///./traffic_monitor.db

# 模型配置
MODEL_PATH=./EfficientNet.pth
CLASS_INDICES_PATH=./class_indices.json
```

### 默认管理员账号

系统会自动创建默认管理员账号：

- **用户名**: `root`
- **密码**: `1234`

⚠️ **安全提示**: 生产环境中请立即修改默认密码！

## 📊 项目结构

```
traffic-recognition-backend/
├── app/                    # 应用主目录
│   ├── api/               # API 路由
│   ├── core/              # 核心配置
│   ├── models/            # 数据模型
│   ├── services/          # 业务逻辑
│   ├── utils/             # 工具函数
│   └── main.py            # FastAPI 应用
├── uploads/               # 上传文件存储
├── cache/                 # 缓存目录
├── EfficientNet.pth       # 训练好的模型文件
├── class_indices.json     # 类别标签映射
├── requirements.txt       # Python 依赖
├── run.py                 # 启动脚本
└── README.md             # 项目文档
```

## 🧪 开发和测试

### 创建测试数据

```bash
python run.py test_data 50  # 创建50条测试记录(但是不建议)
```

### 开发模式启动

```bash
python run.py --no-browser  # 启动时不自动打开浏览器
```

### 运行测试

```bash
pytest tests/              # 运行测试套件
```

## 🚀 部署指南

### Docker 部署

```dockerfile
# Dockerfile 示例
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run.py"]
```

### 生产环境部署

```bash
# 使用 Gunicorn + Uvicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) - 深度学习模型
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📞 支持

由于本项目 vibe coding 含量 99%，如果您遇到任何问题，请：
**自行解决！**

---

⭐ 如果这个项目对您有帮助，请给我一个 star！
