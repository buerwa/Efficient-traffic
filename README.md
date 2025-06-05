# 交通实况识别系统后端

这是一个基于 FastAPI 的交通实况识别系统后端，使用深度学习模型（EfficientNet）来分析交通图像并识别出四种不同的交通状况：事故(accident)、拥堵交通(dense_traffic)、火灾(fire)和畅通交通(sparse_traffic)。

## 主要功能

- 集成 EfficientNet 深度学习模型进行图像识别
- 提供 RESTful API 用于交通图像分析
- 支持用户认证和权限管理
- 提供历史记录查询和统计分析
- 支持实时摄像头图像处理

## 技术栈

- Python 3.9+
- FastAPI 框架
- PyTorch 深度学习框架
- SQLite 数据库
- JWT 认证

## 安装说明

1. 克隆项目仓库
2. 创建并激活 Python 虚拟环境（推荐）
3. 安装依赖包：`pip install -r requirements.txt`
4. 确保 EfficientNet 模型文件在正确位置

## 运行方法

使用以下命令启动后端服务：

```bash
python run.py
```

默认情况下，服务将在 http://localhost:8000 上运行，API 文档可通过 http://localhost:8000/docs 访问。

## API 端点

主要 API 包括：

- `/api/v1/token` - 用户登录和令牌生成
- `/api/v1/predict` - 交通图像分析
- `/api/v1/predictions` - 查询历史分析记录
- `/api/v1/statistics` - 系统统计信息

完整 API 文档请参考 Swagger UI（运行后访问 /docs 路径）。

## 默认管理员账号

系统默认创建管理员账号：

- 用户名: root
- 密码: 1234

可以使用以下命令重新创建管理员账号：

```bash
python create_root_fixed.py
```
