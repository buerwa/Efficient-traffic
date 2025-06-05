from fastapi import APIRouter
from app.api.endpoints import auth, users

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["认证"])
router.include_router(users.router, prefix="/users", tags=["用户管理"])
