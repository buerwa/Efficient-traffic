"""
FastAPI端点定义
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    Request,
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_404_NOT_FOUND,
    HTTP_403_FORBIDDEN,
)
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import os

from app.models.ml_models import ModelManager
from app.models.database import Database
from app.models.schemas import (
    Token,
    UserCreate,
    UserResponse,
    PredictionResponse,
    PredictionsListResponse,
    StatisticsResponse,
    ModelComparisonResponse,
    ErrorResponse,
    PasswordUpdate,
    ProfileUpdate,
    UserStatistics,
    LocationCreate,
    MockDataConfig,
)
from app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    verify_token,
    get_password_hash,
)
from app.utils.helpers import save_uploaded_file, generate_heatmap, compare_predictions
from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, UPLOAD_FOLDER

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# 静态文件服务
router.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 数据库实例
db = Database()
# 模型管理器实例
model_manager = ModelManager()

# 配置模板
templates = Jinja2Templates(directory="app/templates")


# 工具函数: 获取当前用户
async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> Optional[Dict[str, Any]]:
    """根据令牌获取当前用户"""
    token_data = verify_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="无效的令牌或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.get_user_by_username(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# 登录获取令牌
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """登录并获取访问令牌"""
    user = db.get_user_by_username(form_data.username)
    if user is None or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = int((datetime.now() + access_token_expires).timestamp())
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["_id"]},
        expires_delta=access_token_expires,
    )

    # 记录登录事件
    db.add_analytics_event("user_login", {"username": user["username"]}, user["_id"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at,
    }


# 用户注册
@router.post("/users", response_model=UserResponse)
async def register_user(user: UserCreate):
    """注册新用户"""
    # 哈希密码
    hashed_password = hash_password(user.password)

    # 注册用户
    user_id = db.register_user(user.username, user.email, hashed_password)
    if user_id is None:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="用户名或电子邮件已存在",
        )

    # 获取新用户信息
    new_user = db.get_user_by_username(user.username)

    # 记录用户注册事件
    db.add_analytics_event("user_register", {"username": user.username})

    return {
        "id": new_user["_id"],
        "username": new_user["username"],
        "email": new_user["email"],
        "created_at": new_user["created_at"],
    }


# 获取当前用户信息
@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """获取当前用户信息"""
    return {
        "id": current_user["_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"],
    }


# 进行图像预测
@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnet"),  # 默认使用efficientnet
    background_tasks: BackgroundTasks = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """
    对上传的图像进行预测
    """
    try:
        # 强制使用efficientnet模型
        model_name = "efficientnet"

        # 读取图像数据
        image_data = await file.read()

        # 保存上传的文件
        file_path = save_uploaded_file(image_data, file.filename)

        # 使用EfficientNet模型进行预测
        logger.info(f"使用EfficientNet模型进行预测，图像路径: {file_path}")
        prediction = model_manager.predict(image_data, model_name)

        if "error" in prediction:
            logger.error(f"模型预测错误: {prediction['error']}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=prediction["error"]
            )

        # 在后台任务中生成热图
        if background_tasks:
            background_tasks.add_task(generate_heatmap, image_data, prediction)

        # 保存预测结果到数据库
        user_id = current_user["_id"] if current_user else None
        username = current_user["username"] if current_user else None
        prediction_id = db.save_prediction(
            prediction_data=prediction,
            user_id=user_id,
            username=username,
            image_path=file_path,
            image_url=f"/api/v1/uploads/{os.path.basename(file_path)}",
        )

        # 添加分析事件
        db.add_analytics_event(
            "prediction",
            {
                "model": model_name,
                "class": prediction["class_name"],
                "confidence": prediction["confidence"],
            },
            user_id,
        )

        # 添加图像URL至响应
        prediction["image_url"] = f"/api/v1/uploads/{os.path.basename(file_path)}"

        return {
            "id": prediction_id,
            "prediction": prediction,
            "created_at": datetime.now(),
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"预测图像时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测过程中出错: {str(e)}",
        )


# 比较模型预测结果
@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_model_predictions(
    file: UploadFile = File(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """
    使用两个不同的模型对图像进行预测并比较结果
    """
    try:
        # 读取图像数据
        image_data = await file.read()

        # 保存上传的文件
        file_path = save_uploaded_file(image_data, file.filename)

        # 使用两个模型进行预测
        results = model_manager.compare_models(image_data)

        if "error" in results:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=results["error"]
            )

        # 比较预测结果
        comparison = compare_predictions(results["resnet"], results["efficientnet"])

        # 用户ID
        user_id = current_user["_id"] if current_user else None

        # 记录分析事件
        db.add_analytics_event(
            "model_comparison",
            {
                "agreement": comparison["agreement"],
                "resnet_class": results["resnet"]["class_name"],
                "efficientnet_class": results["efficientnet"]["class_name"],
            },
            user_id,
        )

        return {
            "resnet": results["resnet"],
            "efficientnet": results["efficientnet"],
            "recommendation": comparison["recommendation"],
            "agreement": comparison["agreement"],
        }

    except Exception as e:
        logger.error(f"比较模型预测时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"比较过程中出错: {str(e)}",
        )


# 获取预测历史记录
@router.get("/predictions", response_model=PredictionsListResponse)
async def get_prediction_history(
    limit: int = 1000,
    prediction_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取用户的预测历史记录

    参数:
    - limit: 返回记录的最大数量(默认1000，实际获取所有记录)
    - prediction_type: 根据预测类型筛选
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    """
    try:
        logger.info(
            f"获取预测历史记录，用户ID: {current_user['_id']}, 筛选条件: {prediction_type}, {start_date}-{end_date}"
        )

        # 获取所有符合条件的预测记录
        predictions = db.get_predictions(
            user_id=current_user["_id"],
            limit=limit,
            skip=0,
            prediction_type=prediction_type,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"查询到 {len(predictions)} 条预测记录")
        if predictions:
            logger.info(f"第一条记录: {predictions[0]}")

        # 每个记录确保有prediction字段为字符串
        for pred in predictions:
            if isinstance(pred.get("prediction"), dict):
                # 如果prediction是一个对象，提取主要预测结果
                if "class_name" in pred["prediction"]:
                    pred["prediction"] = pred["prediction"]["class_name"]
                elif "prediction" in pred["prediction"]:
                    pred["prediction"] = pred["prediction"]["prediction"]
            # 确保_id字段存在
            if "_id" not in pred and "id" in pred:
                pred["_id"] = pred["id"]

        result = {
            "items": predictions,  # 新的响应格式
            "predictions": predictions,  # 兼容旧的响应格式
            "total": len(predictions),
            "count": len(predictions),  # 兼容旧的响应格式
            "limit": limit,
        }

        logger.info(f"返回结果格式: {result.keys()}")
        return result

    except Exception as e:
        logger.error(f"获取预测历史记录时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取历史记录时出错: {str(e)}",
        )


# 获取系统统计信息
@router.get("/statistics", response_model=StatisticsResponse)
async def get_system_statistics(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取系统统计信息
    """
    try:
        stats = db.get_statistics()

        if "error" in stats:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=stats["error"]
            )

        return stats

    except Exception as e:
        logger.error(f"获取统计信息时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息时出错: {str(e)}",
        )


# 健康检查端点
@router.get("/health")
async def health_check():
    """
    API健康检查
    """
    return {
        "status": "healthy",
        "service": "FastAPI",
        "timestamp": datetime.now().isoformat(),
    }


# 获取上传的文件
@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """获取上传的文件"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)


# 首页
@router.get("/")
async def homepage():
    """返回系统主页"""
    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>交通实况识别系统 API</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #1677ff; }}
                a {{ color: #1677ff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                .endpoint {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                code {{ background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>交通实况识别系统 API</h1>
            <p>欢迎使用交通实况识别系统后端API。本系统使用EfficientNet深度学习模型进行交通图像分析。</p>
            
            <h2>API文档</h2>
            <p>请访问以下链接查看完整的API文档：</p>
            <ul>
                <li><a href="/docs" target="_blank">Swagger文档</a></li>
                <li><a href="/redoc" target="_blank">ReDoc文档</a></li>
            </ul>
            
            <h2>接口示例</h2>
            <div class="endpoint">
                <p><strong>图像识别：</strong> <code>POST /api/v1/predict</code></p>
                <p>上传图像进行交通实况识别</p>
            </div>
            
            <div class="endpoint">
                <p><strong>历史记录：</strong> <code>GET /api/v1/predictions</code></p>
                <p>获取历史识别记录</p>
            </div>
            
            <div class="endpoint">
                <p><strong>用户管理：</strong> <code>GET /api/v1/users/me</code></p>
                <p>获取当前用户信息</p>
            </div>
            
            <h2>状态</h2>
            <p>服务器运行正常</p>
        </body>
        </html>
        """,
        media_type="text/html",
    )


# 添加用户密码修改端点
@router.put("/users/password", response_model=Dict[str, str])
async def update_password(
    password_update: PasswordUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    更新当前用户的密码
    """
    try:
        # 验证当前密码
        stored_user = db.get_user_by_username(current_user["username"])
        if not stored_user:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="用户不存在")

        # 验证当前密码
        if not verify_password(
            password_update.current_password, stored_user["password"]
        ):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="当前密码不正确"
            )

        # 哈希新密码
        hashed_password = get_password_hash(password_update.new_password)

        # 更新密码
        success = db.update_user_password(current_user["_id"], hashed_password)
        if not success:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="密码更新失败"
            )

        return {"message": "密码更新成功"}

    except Exception as e:
        logger.error(f"更新密码时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新密码时出错: {str(e)}",
        )


# 添加用户信息更新端点
@router.put("/users/profile", response_model=UserResponse)
async def update_profile(
    profile_update: ProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    更新当前用户的个人资料
    """
    try:
        # 更新用户信息
        updated_user = db.update_user_profile(
            current_user["_id"], profile_update.dict()
        )
        if not updated_user:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="更新个人资料失败"
            )

        return updated_user

    except Exception as e:
        logger.error(f"更新个人资料时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新个人资料时出错: {str(e)}",
        )


# 获取预设地点列表
@router.get("/locations", response_model=List[Dict[str, str]])
async def get_locations(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """
    获取预设地点列表
    """
    try:
        locations = db.get_locations()
        return locations
    except Exception as e:
        logger.error(f"获取地点列表时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取地点列表时出错: {str(e)}",
        )


# 添加新地点
@router.post("/locations", response_model=Dict[str, str])
async def add_location(
    location_data: LocationCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    添加新的预设地点
    需要管理员权限
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="需要管理员权限")

    try:
        location_id = db.add_location(location_data.dict())
        return {"id": location_id, "message": "地点添加成功"}
    except Exception as e:
        logger.error(f"添加地点时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加地点时出错: {str(e)}",
        )


# 生成模拟的交通实况历史数据
@router.post("/generate-mock-data", response_model=Dict[str, str])
async def generate_mock_data(
    config: MockDataConfig,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    生成模拟的交通实况历史数据
    仅管理员可访问
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="需要管理员权限")

    try:
        count = db.generate_mock_prediction_data(
            count=config.count,
            days_range=config.days_range,
            user_id=current_user["_id"],
        )
        return {"message": f"成功生成{count}条模拟数据"}
    except Exception as e:
        logger.error(f"生成模拟数据时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成模拟数据时出错: {str(e)}",
        )


# 获取当前用户的数据统计
@router.get("/users/statistics", response_model=UserStatistics)
async def get_user_statistics(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取当前用户的数据统计信息
    """
    try:
        stats = db.get_user_statistics(current_user["_id"])
        return stats
    except Exception as e:
        logger.error(f"获取用户统计信息时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户统计信息时出错: {str(e)}",
        )


# 上传用户头像
@router.post("/upload-avatar", response_model=Dict[str, str])
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    上传并更新用户头像
    """
    try:
        # 检查文件是否为图片
        content_type = file.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="文件必须是图片类型",
            )

        # 读取图像数据
        image_data = await file.read()

        # 生成唯一的文件名
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        avatar_filename = f"avatar_{current_user['_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_extension}"

        # 保存上传的文件
        file_path = save_uploaded_file(image_data, avatar_filename)
        if not file_path:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="保存头像文件失败",
            )

        # 更新用户记录中的头像URL
        avatar_url = f"/api/v1/uploads/{os.path.basename(file_path)}"
        profile_update = {"avatar_url": avatar_url}

        updated_user = db.update_user_profile(current_user["_id"], profile_update)
        if not updated_user:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="更新用户头像失败",
            )

        # 记录分析事件
        db.add_analytics_event(
            "avatar_upload",
            {"username": current_user["username"]},
            current_user["_id"],
        )

        return {
            "url": avatar_url,
            "message": "头像上传成功",
        }

    except Exception as e:
        logger.error(f"上传头像时出错: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传头像时出错: {str(e)}",
        )
