"""
安全性工具函数：JWT令牌、密码哈希等
"""

from datetime import datetime, timedelta
from typing import Union, Any, Optional, Dict
import bcrypt
from jose import JWTError, jwt
import logging
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from starlette.status import HTTP_401_UNAUTHORIZED
from passlib.context import CryptContext

from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.models.schemas import TokenData
from app.models.database import Database

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2密码Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

# 数据库实例
db = Database()


def hash_password(password: str) -> str:
    """
    对密码进行哈希处理

    Args:
        password: 明文密码

    Returns:
        str: 哈希密码
    """
    # 使用简单的SHA-256哈希
    import hashlib

    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否正确

    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码

    Returns:
        bool: 密码是否匹配
    """
    try:
        # 直接使用简单哈希验证（SHA-256）
        import hashlib

        simple_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        result = simple_hash == hashed_password

        logger.info(f"密码验证结果: {result}")
        return result

    except Exception as e:
        logger.error(f"密码验证失败: {e}")
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌

    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量（可选）

    Returns:
        str: JWT令牌
    """
    to_encode = data.copy()

    # 设置过期时间
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    # 编码JWT
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    验证JWT令牌并解码

    Args:
        token: JWT令牌

    Returns:
        TokenData: 从令牌中解码的数据，如果无效则为None
    """
    try:
        # 解码JWT
        # 同时支持HS256和ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM, "HS256"])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        exp = payload.get("exp")

        logger.info(f"验证令牌，用户: {username}, 用户ID: {user_id}")

        if username is None:
            return None

        return TokenData(username=username, user_id=user_id, exp=exp)

    except JWTError as e:
        logger.warning(f"无效的令牌: {e}")
        return None
    except Exception as e:
        logger.error(f"验证令牌时出错: {e}")
        return None


def get_password_hash(password: str) -> str:
    """
    获取密码哈希

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码
    """
    return pwd_context.hash(password)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    获取当前用户

    Args:
        token: JWT令牌

    Returns:
        Dict: 用户信息

    Raises:
        HTTPException: 如果令牌无效或过期
    """
    credentials_exception = HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="凭据无效",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 使用verify_token函数解析token
        logger.info(f"尝试验证令牌并获取用户信息")
        token_data = verify_token(token)

        if token_data is None:
            logger.warning("令牌验证失败，token_data为None")
            raise credentials_exception

        if token_data.username is None:
            logger.warning("令牌验证失败，username为None")
            raise credentials_exception

        # 获取用户名
        username = token_data.username
        logger.info(f"令牌验证成功，用户名: {username}")

    except Exception as e:
        logger.error(f"验证令牌时出错: {e}")
        raise credentials_exception

    # 获取用户信息
    user = db.get_user_by_username(username)
    if user is None:
        logger.warning(f"用户不存在: {username}")
        raise credentials_exception

    logger.info(f"成功获取用户信息: {username}")
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[Dict[str, Any]]:
    """
    获取当前用户（可选）

    Args:
        token: JWT令牌

    Returns:
        Dict or None: 用户信息或None
    """
    if not token:
        return None

    try:
        user = await get_current_user(token)
        return user
    except HTTPException:
        return None


# 重命名第二个verify_token函数为verify_token_payload
def verify_token_payload(token: str) -> Optional[Dict[str, Any]]:
    """
    验证令牌并返回原始payload

    Args:
        token: JWT令牌

    Returns:
        Dict or None: 令牌数据或None
    """
    try:
        # 同时支持HS256和ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM, "HS256"])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"验证令牌payload失败: {e}")
        return None
