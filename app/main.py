"""
主应用程序文件
FastAPI 应用程序
"""

import logging
import os
import sys
import uvicorn
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
import jwt
from passlib.context import CryptContext
import shutil
from pathlib import Path
import re
import uuid
import json
from fastapi.staticfiles import StaticFiles

from app.core.config import (
    API_V1_STR,
    FASTAPI_PORT,
    DEBUG,
    MAX_CONTENT_LENGTH,
    UPLOAD_FOLDER,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    MODEL_PATH,
    CLASS_INDICES_PATH,
)

from app.services.image_recognition import EfficientNetModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化图像识别模型
try:
    image_model = EfficientNetModel(MODEL_PATH, CLASS_INDICES_PATH)
    logger.info("图像识别模型加载成功")
except Exception as e:
    logger.error(f"图像识别模型加载失败: {str(e)}")
    raise

# 认证相关
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{API_V1_STR}/token")


class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    role: str = "user"
    is_active: bool = True


class UserInDB(User):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user(username: str):
    from app.models.database import Database

    db = Database()
    user = db.get_user_by_username(username)
    if user:
        return UserInDB(**user)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # 默认令牌有效期增加到24小时
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except (jwt.exceptions.PyJWTError, jwt.exceptions.ExpiredSignatureError):
        # 修复JWT异常处理
        logger.warning("JWT验证失败: 令牌无效或已过期")
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    return current_user


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="交通实况识别系统 API",
        description="集成了EfficientNet深度学习模型的交通实况识别系统 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 为上传文件夹提供静态文件服务
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

    # 添加调试信息
    print(f"已挂载静态文件目录: {UPLOAD_FOLDER} -> /uploads")
    print(f"静态文件目录绝对路径: {os.path.abspath(UPLOAD_FOLDER)}")

    # 认证路由
    @app.post(f"{API_V1_STR}/token", response_model=Token)
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    @app.post(f"{API_V1_STR}/register")
    async def register_user(
        username: str = Form(...),
        password: str = Form(...),
        full_name: Optional[str] = Form(None),
    ):
        """注册新用户

        Args:
            username: 用户名
            password: 密码
            full_name: 可选的全名

        Returns:
            成功消息

        Raises:
            HTTPException: 如果用户名已存在或密码不符合要求
        """
        try:
            # 打印请求参数便于调试
            logger.info(f"尝试注册新用户: username={username}, full_name={full_name}")

            # 验证密码长度
            if len(password) < 6:
                logger.warning(f"用户 {username} 注册失败: 密码太短")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="密码长度必须大于6个字符",
                )

            # 验证用户名格式
            if not re.match("^[a-zA-Z0-9_-]+$", username):
                logger.warning(f"用户 {username} 注册失败: 用户名格式无效")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="用户名只能包含字母、数字、下划线和连字符",
                )

            from app.models.database import Database

            db = Database()
            if db.get_user_by_username(username):
                logger.warning(f"用户 {username} 注册失败: 用户名已存在")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在"
                )

            hashed_password = get_password_hash(password)
            user = {
                "username": username,
                "password": hashed_password,
                "full_name": full_name,
                "role": "user",
            }

            created_user = db.create_user(**user)
            logger.info(f"用户 {username} 注册成功")
            return {"message": "用户注册成功", "user_id": created_user["_id"]}

        except ValueError as e:
            # 捕获数据库中的验证错误
            logger.error(f"用户 {username} 注册失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )
        except Exception as e:
            # 捕获所有其他错误
            logger.error(f"用户 {username} 注册时发生未知错误: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误，请稍后再试",
            )

    @app.get(f"{API_V1_STR}/users/me", response_model=User)
    async def read_users_me(current_user: User = Depends(get_current_active_user)):
        return current_user

    # 用户管理路由
    @app.get(f"{API_V1_STR}/users", response_model=list[User])
    async def read_users(
        search: Optional[str] = None,
        current_user: User = Depends(get_current_active_user),
    ):
        """获取用户列表

        Args:
            search: 搜索关键词，可按用户名或全名搜索
            current_user: 当前登录用户

        Returns:
            用户列表
        """
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="没有权限访问此资源"
            )
        from app.models.database import Database

        db = Database()
        return db.get_all_users(search_term=search)

    @app.get(f"{API_V1_STR}/users/{{username}}", response_model=User)
    async def read_user(
        username: str, current_user: User = Depends(get_current_active_user)
    ):
        if current_user.role != "admin" and current_user.username != username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="没有权限访问此资源"
            )
        from app.models.database import Database

        db = Database()
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在"
            )
        return user

    @app.put(f"{API_V1_STR}/users/{{username}}")
    async def update_user(
        username: str,
        user_update: dict,
        current_user: User = Depends(get_current_active_user),
    ):
        if current_user.role != "admin" and current_user.username != username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="没有权限修改此用户"
            )

        from app.models.database import Database

        db = Database()
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在"
            )

        update_data = {k: v for k, v in user_update.items() if v is not None}
        if "password" in update_data:
            update_data["password"] = get_password_hash(update_data["password"])

        db.update_user(user["_id"], **update_data)
        return {"message": "用户更新成功"}

    @app.delete(f"{API_V1_STR}/users/{{username}}")
    async def delete_user(
        username: str, current_user: User = Depends(get_current_active_user)
    ):
        if current_user.role != "admin" and current_user.username != username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="没有权限删除此用户"
            )

        from app.models.database import Database

        db = Database()
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在"
            )

        db.delete_user(user["_id"])
        return {"message": "用户删除成功"}

    # 图像识别路由
    @app.post(f"{API_V1_STR}/predict")
    async def predict_traffic(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_active_user),
    ):
        """
        对上传的交通图片进行预测分析

        Args:
            file: 上传的图片文件
            current_user: 当前登录用户

        Returns:
            预测结果，包含预测类别和置信度

        Raises:
            HTTPException: 当图片处理或预测失败时
        """
        try:
            # 验证文件类型
            filename = file.filename
            if not filename:
                logger.error("文件名为空")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="上传的文件名不能为空",
                )

            # 检查文件扩展名
            file_ext = filename.split(".")[-1].lower()
            if file_ext not in ["jpg", "jpeg", "png"]:
                logger.error(f"不支持的文件类型: {file_ext}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="只支持JPG和PNG格式的图片",
                )

            # 确保上传目录存在
            upload_dir = os.path.join(UPLOAD_FOLDER, current_user.username)
            os.makedirs(upload_dir, exist_ok=True)

            # 生成唯一文件名，避免覆盖
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(upload_dir, unique_filename)

            # 保存上传的文件
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                logger.info(
                    f"用户 {current_user.username} 上传图片 {filename} 已保存到 {file_path}"
                )
            except Exception as save_error:
                logger.error(f"保存文件失败: {str(save_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存文件失败: {str(save_error)}",
                )

            # 进行预测
            try:
                result = image_model.predict(file_path)

                # 确保result中的prediction字段是标准类型之一
                prediction_raw = result.get("prediction", "")
                logger.info(f"原始预测类型: {prediction_raw}")

                # 英文类型映射到中文
                type_mappings = {
                    "fire": "车辆起火",
                    "sparse_traffic": "交通顺畅",
                    "accident": "交通事故",
                    "dense_traffic": "交通拥堵",
                }

                # 映射预测类型
                result["prediction"] = type_mappings.get(prediction_raw, prediction_raw)
                if result["prediction"] not in [
                    "车辆起火",
                    "交通顺畅",
                    "交通事故",
                    "交通拥堵",
                ]:
                    logger.warning(
                        f"未知的预测类型: {prediction_raw} -> 设为默认值: 交通事故"
                    )
                    result["prediction"] = "交通事故"

                # 记录预测结果
                logger.info(
                    f"用户 {current_user.username} 上传图片 {filename} 的预测结果: {result}"
                )

                # 添加到数据库中
                try:
                    from app.models.database import Database

                    db = Database()

                    # 创建当前时间戳
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    prediction_id = db.add_prediction(
                        user_id=getattr(current_user, "_id", None),
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        image_path=file_path,
                        username=current_user.username,
                        model_name="efficientnet",
                        timestamp=current_timestamp,
                        labels=json.dumps(
                            ["traffic", result.get("prediction", "accident")]
                        ),
                        filename=unique_filename,
                    )
                    logger.info(f"预测结果已保存到数据库，ID: {prediction_id}")

                    # 添加图片URL到结果中
                    image_url = f"/uploads/{current_user.username}/{unique_filename}"
                    result["image_url"] = image_url
                    result["id"] = prediction_id

                    # 更新数据库中的image_url字段
                    try:
                        db.conn.cursor().execute(
                            "UPDATE predictions SET image_url = ? WHERE _id = ?",
                            (image_url, prediction_id),
                        )
                        db.conn.commit()
                        logger.info(
                            f"已更新预测记录 {prediction_id} 的image_url: {image_url}"
                        )
                    except Exception as url_error:
                        logger.error(f"更新image_url失败: {str(url_error)}")
                except Exception as db_error:
                    # 数据库保存失败，但不影响返回结果
                    logger.error(f"保存预测结果到数据库失败: {str(db_error)}")

                return result

            except Exception as pred_error:
                logger.error(f"预测模型执行失败: {str(pred_error)}")
                error_message = str(pred_error)
                # 提供更详细的错误信息
                if "CUDA out of memory" in error_message:
                    error_message = "GPU内存不足，请稍后重试或使用较小的图片"
                elif "Expected 3-channel" in error_message:
                    error_message = "图片格式异常，请使用标准的RGB彩色图片"
                elif (
                    "not found" in error_message.lower()
                    or "no such file" in error_message.lower()
                ):
                    error_message = "模型文件或图片文件未找到，请联系管理员"

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": "图像分析失败", "message": error_message},
                )

        except HTTPException:
            # 直接重新抛出HTTP异常
            raise
        except Exception as e:
            logger.error(f"预测过程中发生未知错误: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"处理图片时发生错误: {str(e)}",
            )

    # 获取历史预测记录
    @app.get(f"{API_V1_STR}/predictions")
    async def get_predictions(
        limit: int = 20,
        prediction_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        current_user: User = Depends(get_current_active_user),
    ):
        """获取预测历史记录

        Args:
            limit: 返回的记录数量限制
            prediction_type: 根据预测类型筛选（可选）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            current_user: 当前登录用户

        Returns:
            预测记录列表
        """
        try:
            # 导入数据库模块
            from app.models.database import Database

            db = Database()

            # 根据用户角色决定获取所有记录还是仅获取用户自己的记录
            if current_user.role == "admin":
                # 管理员可以看到所有记录
                predictions = db.get_all_predictions(
                    limit=limit,
                    prediction_type=prediction_type,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                # 普通用户只能看到自己的记录
                user_id = None
                try:
                    # 尝试从用户对象中获取ID
                    user = db.get_user_by_username(current_user.username)
                    if user:
                        user_id = user["_id"]
                except Exception as e:
                    logger.error(f"获取用户ID失败: {str(e)}")

                if user_id:
                    predictions = db.get_user_predictions(
                        user_id=user_id,
                        limit=limit,
                        prediction_type=prediction_type,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    # 没有用户ID，尝试通过用户名查找
                    predictions = []
                    logger.warning(
                        f"用户 {current_user.username} 没有对应的ID，无法获取预测记录"
                    )

            # 定义允许的标准类型及其映射
            standard_types = {
                "交通事故": "交通事故",
                "交通顺畅": "交通顺畅",
                "交通拥堵": "交通拥堵",
                "车辆起火": "车辆起火",
            }

            # 英文到中文的映射
            english_to_chinese = {
                "sparse_traffic": "交通顺畅",
                "dense_traffic": "交通拥堵",
                "accident": "交通事故",
                "fire": "车辆起火",
                # 兼容其他可能的英文格式
                "traffic accident": "交通事故",
                "smooth traffic": "交通顺畅",
                "traffic congestion": "交通拥堵",
                "vehicle fire": "车辆起火",
            }

            result = []
            for pred in predictions:
                pred_dict = dict(pred)
                # 尝试解析prediction字段
                try:
                    prediction_text = pred_dict.get("prediction", "")
                    logger.debug(f"原始预测: {prediction_text}")

                    # 优先检查是否是英文类型（模型直接输出）
                    if prediction_text in english_to_chinese:
                        pred_dict["prediction"] = english_to_chinese[prediction_text]

                    # 如果prediction已经是标准中文类型之一
                    elif prediction_text in standard_types:
                        pred_dict["prediction"] = standard_types[prediction_text]

                    # 尝试将英文映射为中文（小写匹配）
                    elif prediction_text.lower() in english_to_chinese:
                        pred_dict["prediction"] = english_to_chinese[
                            prediction_text.lower()
                        ]

                    # 尝试解析JSON结构
                    elif "{" in prediction_text:
                        try:
                            pred_obj = json.loads(prediction_text)
                            if isinstance(pred_obj, dict):
                                if "prediction" in pred_obj:
                                    class_name = pred_obj["prediction"]
                                elif "class_name" in pred_obj:
                                    class_name = pred_obj["class_name"]
                                else:
                                    class_name = None

                                if class_name and class_name in english_to_chinese:
                                    pred_dict["prediction"] = english_to_chinese[
                                        class_name
                                    ]
                                elif (
                                    class_name
                                    and class_name.lower() in english_to_chinese
                                ):
                                    pred_dict["prediction"] = english_to_chinese[
                                        class_name.lower()
                                    ]
                                else:
                                    # 默认为交通事故
                                    pred_dict["prediction"] = "交通事故"
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            # JSON解析失败，使用默认值
                            pred_dict["prediction"] = "交通事故"

                    # 其他情况默认为交通事故
                    else:
                        pred_dict["prediction"] = "交通事故"

                    logger.debug(f"映射后预测: {pred_dict['prediction']}")
                except Exception as e:
                    logger.error(f"处理预测类型时出错: {str(e)}")
                    pred_dict["prediction"] = "交通事故"  # 出错时默认为交通事故

                # 添加到结果列表
                result.append(pred_dict)

            return {
                "items": result,  # 前端主要期望的字段
                "predictions": result,  # 保持向后兼容
                "total": len(result),
                "count": len(result),  # 保持向后兼容
                "limit": limit,
            }
        except Exception as e:
            logger.error(f"获取预测历史记录失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取预测历史记录失败: {str(e)}",
            )

    # 交通数据分析接口
    @app.get(f"{API_V1_STR}/statistics")
    async def get_traffic_statistics(
        current_user: User = Depends(get_current_active_user),
    ):
        """获取交通数据统计分析

        Args:
            current_user: 当前登录用户

        Returns:
            Dict: 包含各类统计数据
        """
        try:
            from app.models.database import Database

            db = Database()

            # 获取统计数据
            statistics = db.get_prediction_statistics()

            return statistics
        except Exception as e:
            logger.error(f"获取交通数据统计分析失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取交通数据统计分析失败: {str(e)}",
            )

    @app.get(f"{API_V1_STR}/confidence_trend")
    async def get_confidence_trend(
        days: int = 30,
        count: int = 100,
        current_user: User = Depends(get_current_active_user),
    ):
        """获取置信度趋势数据

        Args:
            days: 天数范围（已废弃，保留参数仅为兼容）
            count: 获取最近多少条记录的趋势
            current_user: 当前登录用户

        Returns:
            List[Dict]: 预测记录ID和置信度数据列表
        """
        try:
            from app.models.database import Database

            db = Database()

            # 获取置信度趋势数据
            trend_data = db.get_confidence_trend(days=days, count=count)

            return trend_data
        except Exception as e:
            logger.error(f"获取置信度趋势数据失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取置信度趋势数据失败: {str(e)}",
            )

    @app.delete(f"{API_V1_STR}/predictions/{{prediction_id}}")
    async def delete_prediction(
        prediction_id: str,
        current_user: User = Depends(get_current_active_user),
    ):
        """删除预测记录

        Args:
            prediction_id: 预测记录ID
            current_user: 当前登录用户

        Returns:
            删除成功的消息

        Raises:
            HTTPException: 当预测记录不存在或无权删除时
        """
        try:
            from app.models.database import Database

            db = Database()

            # 获取预测记录，检查所有权
            # 如果是管理员可以删除任何记录，普通用户只能删除自己的记录
            if current_user.role != "admin":
                # 获取用户ID
                user = db.get_user_by_username(current_user.username)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="用户不存在",
                    )

                # 查询记录是否属于当前用户
                cursor = db.conn.cursor()
                cursor.execute(
                    "SELECT user_id FROM predictions WHERE _id = ?", (prediction_id,)
                )
                record = cursor.fetchone()

                if not record:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="预测记录不存在",
                    )

                if record["user_id"] != user["_id"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="无权删除该预测记录",
                    )

            # 执行删除操作
            success = db.delete_prediction(prediction_id)

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="预测记录不存在或删除失败",
                )

            return {"message": "预测记录已成功删除"}

        except HTTPException:
            # 直接重新抛出HTTP异常
            raise
        except Exception as e:
            logger.error(f"删除预测记录时出错: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"删除预测记录时出错: {str(e)}",
            )

    # 添加专门用于返回中文预测结果的路由
    @app.get(f"{API_V1_STR}/chinese_predictions")
    async def get_chinese_predictions(
        limit: int = 500,
        prediction_type: Optional[str] = None,
        current_user: User = Depends(get_current_active_user),
    ):
        """获取预测结果并将其转换为标准中文类型

        Args:
            limit: 返回的记录数量限制
            prediction_type: 根据预测类型筛选（可选）
            current_user: 当前登录用户

        Returns:
            包含标准化中文预测类型的预测记录列表
        """
        try:
            from app.models.database import Database

            db = Database()

            # 根据用户角色获取预测记录
            if current_user.role == "admin":
                predictions = db.get_all_predictions(
                    limit=limit, prediction_type=prediction_type
                )
            else:
                user = db.get_user_by_username(current_user.username)
                if user:
                    predictions = db.get_user_predictions(
                        user_id=user["_id"],
                        limit=limit,
                        prediction_type=prediction_type,
                    )
                else:
                    logger.warning(
                        f"用户 {current_user.username} 没有对应的ID，无法获取预测记录"
                    )
                    predictions = []

            # 定义允许的标准类型及其映射
            standard_types = {
                "交通事故": "交通事故",
                "交通顺畅": "交通顺畅",
                "交通拥堵": "交通拥堵",
                "车辆起火": "车辆起火",
            }

            # 英文到中文的映射
            english_to_chinese = {
                "sparse_traffic": "交通顺畅",
                "dense_traffic": "交通拥堵",
                "accident": "交通事故",
                "fire": "车辆起火",
                # 兼容其他可能的英文格式
                "traffic accident": "交通事故",
                "smooth traffic": "交通顺畅",
                "traffic congestion": "交通拥堵",
                "vehicle fire": "车辆起火",
            }

            result = []
            for pred in predictions:
                pred_dict = dict(pred)
                # 尝试解析prediction字段
                try:
                    prediction_text = pred_dict.get("prediction", "")
                    logger.debug(f"原始预测: {prediction_text}")

                    # 优先检查是否是英文类型（模型直接输出）
                    if prediction_text in english_to_chinese:
                        pred_dict["prediction"] = english_to_chinese[prediction_text]

                    # 如果prediction已经是标准中文类型之一
                    elif prediction_text in standard_types:
                        pred_dict["prediction"] = standard_types[prediction_text]

                    # 尝试将英文映射为中文（小写匹配）
                    elif prediction_text.lower() in english_to_chinese:
                        pred_dict["prediction"] = english_to_chinese[
                            prediction_text.lower()
                        ]

                    # 尝试解析JSON结构
                    elif "{" in prediction_text:
                        try:
                            pred_obj = json.loads(prediction_text)
                            if isinstance(pred_obj, dict):
                                if "prediction" in pred_obj:
                                    class_name = pred_obj["prediction"]
                                elif "class_name" in pred_obj:
                                    class_name = pred_obj["class_name"]
                                else:
                                    class_name = None

                                if class_name and class_name in english_to_chinese:
                                    pred_dict["prediction"] = english_to_chinese[
                                        class_name
                                    ]
                                elif (
                                    class_name
                                    and class_name.lower() in english_to_chinese
                                ):
                                    pred_dict["prediction"] = english_to_chinese[
                                        class_name.lower()
                                    ]
                                else:
                                    # 默认为交通事故
                                    pred_dict["prediction"] = "交通事故"
                        except (json.JSONDecodeError, AttributeError, TypeError):
                            # JSON解析失败，使用默认值
                            pred_dict["prediction"] = "交通事故"

                    # 其他情况默认为交通事故
                    else:
                        pred_dict["prediction"] = "交通事故"

                    logger.debug(f"映射后预测: {pred_dict['prediction']}")
                except Exception as e:
                    logger.error(f"处理预测类型时出错: {str(e)}")
                    pred_dict["prediction"] = "交通事故"  # 出错时默认为交通事故

                result.append(pred_dict)

            return {"data": result, "count": len(result)}

        except Exception as e:
            logger.error(f"获取中文预测记录失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"获取中文预测记录失败: {str(e)}",
            )

    # 批量删除预测记录
    @app.post(f"{API_V1_STR}/predictions/batch-delete")
    async def batch_delete_predictions(
        ids: list = Body(...),
        current_user: User = Depends(get_current_active_user),
    ):
        """批量删除预测记录

        Args:
            ids: 要删除的预测记录ID列表
            current_user: 当前登录用户

        Returns:
            删除成功的消息
        """
        try:
            from app.models.database import Database

            db = Database()

            # 管理员可以删除任何记录
            if current_user.role == "admin":
                deleted_count = 0
                for prediction_id in ids:
                    success = db.delete_prediction(prediction_id)
                    if success:
                        deleted_count += 1

                return {
                    "message": f"成功删除{deleted_count}/{len(ids)}条预测记录",
                    "deleted_count": deleted_count,
                }
            else:
                # 普通用户只能删除自己的记录
                user = db.get_user_by_username(current_user.username)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="用户不存在",
                    )

                # 获取用户有权删除的记录
                cursor = db.conn.cursor()
                deleted_count = 0
                unauthorized_count = 0

                for prediction_id in ids:
                    # 检查记录所有权
                    cursor.execute(
                        "SELECT user_id FROM predictions WHERE _id = ?",
                        (prediction_id,),
                    )
                    record = cursor.fetchone()

                    if not record:
                        continue  # 记录不存在，跳过

                    if record["user_id"] == user["_id"]:
                        # 用户有权删除
                        success = db.delete_prediction(prediction_id)
                        if success:
                            deleted_count += 1
                    else:
                        unauthorized_count += 1

                return {
                    "message": f"成功删除{deleted_count}/{len(ids)}条预测记录，{unauthorized_count}条无权删除",
                    "deleted_count": deleted_count,
                    "unauthorized_count": unauthorized_count,
                }

        except Exception as e:
            logger.error(f"批量删除预测记录失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"批量删除预测记录失败: {str(e)}",
            )

    @app.get("/")
    async def root():
        """API根路径返回欢迎信息"""
        return {
            "message": "欢迎使用交通实况识别系统 API",
            "documentation": "/docs",
            "redoc": "/redoc",
        }

    logger.info("FastAPI 应用创建成功")
    return app


def init_admin():
    """初始化管理员账户"""
    from app.models.database import Database

    db = Database()
    admin = db.get_user_by_username("root")
    if not admin:
        hashed_password = get_password_hash("1234")
        db.create_user(
            username="root",
            password=hashed_password,
            full_name="系统管理员",
            role="admin",
        )
        logger.info("管理员账户创建成功")
    else:
        logger.info("管理员账户已存在")


def run_server():
    """运行 FastAPI 服务器"""
    app = create_app()
    init_admin()  # 初始化管理员账户
    logger.info(f"启动 FastAPI 服务器在端口 {FASTAPI_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT, log_level="info")


if __name__ == "__main__":
    run_server()
