"""
Pydantic 数据模型/模式
用于请求和响应的数据验证
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re


class Token(BaseModel):
    """令牌响应模型"""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """令牌数据模型"""

    username: Optional[str] = None
    user_id: Optional[str] = None
    exp: Optional[int] = None


class UserBase(BaseModel):
    """用户基本信息"""

    username: str


class UserCreate(UserBase):
    """创建用户请求模型"""

    password: str
    full_name: Optional[str] = None

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("密码长度必须大于6个字符")
        return v

    @validator("username")
    def username_alphanumeric(cls, v):
        if not re.match("^[a-zA-Z0-9_-]+$", v):
            raise ValueError("用户名只能包含字母、数字、下划线和连字符")
        return v


class UserLogin(BaseModel):
    """用户登录请求模型"""

    username: str
    password: str
    remember_me: bool = False


class UserResponse(UserBase):
    """用户响应模型"""

    _id: str
    full_name: Optional[str] = None
    created_at: str
    is_active: bool = True
    role: str = "user"
    avatar_url: Optional[str] = None


class CurrentUser(UserResponse):
    """当前用户模型"""

    pass


# 密码更新模型
class PasswordUpdate(BaseModel):
    """密码更新请求模型"""

    current_password: str
    new_password: str

    @validator("new_password")
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("新密码长度必须大于6个字符")
        return v


# 用户资料更新模型
class ProfileUpdate(BaseModel):
    """用户资料更新模型"""

    full_name: Optional[str] = None
    avatar_url: Optional[str] = None


class PredictionBase(BaseModel):
    """预测基本信息"""

    class_name: str
    class_id: int
    confidence: float
    all_probabilities: Dict[str, float]
    model_used: str
    processing_time_ms: float


class PredictionCreate(BaseModel):
    """创建预测请求模型"""

    model_name: Optional[str] = "efficientnet"
    location_id: Optional[str] = None
    weather: Optional[str] = None


class PredictionResponse(BaseModel):
    """预测响应模型"""

    id: str
    prediction: PredictionBase
    created_at: datetime
    user_id: Optional[str] = None
    location: Optional[str] = None
    weather: Optional[str] = None


class PredictionsListResponse(BaseModel):
    """预测列表响应模型"""

    items: List[Dict[str, Any]]  # 允许任何类型的字段
    predictions: Optional[List[Dict[str, Any]]] = None  # 兼容旧格式
    total: int = 0
    count: Optional[int] = None  # 兼容旧格式
    page: int = 1
    limit: int = 10


class StatisticsResponse(BaseModel):
    """统计信息响应模型"""

    total_predictions: int
    total_users: int
    class_distribution: Dict[str, int]
    model_usage: Dict[str, int]


class ModelComparisonResponse(BaseModel):
    """模型比较响应模型"""

    resnet: Optional[PredictionBase] = None
    efficientnet: Optional[PredictionBase] = None
    recommendation: str
    agreement: bool


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error: str
    detail: Optional[str] = None


# 用户统计数据模型
class UserStatistics(BaseModel):
    """用户统计数据"""

    total_predictions: int
    recent_predictions: int
    favorite_location: Optional[str] = None
    most_detected_class: Optional[str] = None
    class_distribution: Dict[str, int]
    prediction_by_day: Dict[str, int]


# 地点模型
class LocationBase(BaseModel):
    """地点基本信息"""

    name: str
    city: str
    province: Optional[str] = None
    description: Optional[str] = None


class LocationCreate(LocationBase):
    """创建地点请求模型"""

    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LocationResponse(LocationBase):
    """地点响应模型"""

    id: str
    created_at: datetime


# 模拟数据生成配置
class MockDataConfig(BaseModel):
    """模拟数据生成配置"""

    count: int = Field(default=20, ge=1, le=100)
    days_range: int = Field(default=30, ge=1, le=365)
    include_weekends: bool = True
    random_seed: Optional[int] = None


# 天气状况枚举（非Enum，用于前端显示）
class WeatherOptions:
    """天气状况选项"""

    SUNNY = "晴天"
    CLOUDY = "多云"
    RAINY = "雨天"
    SNOWY = "雪天"
    FOGGY = "雾天"
    ALL = ["晴天", "多云", "雨天", "雪天", "雾天"]
