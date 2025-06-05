from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from pydantic import BaseModel

from app.models.database import Database
from app.api.endpoints.auth import get_current_active_user, User

router = APIRouter()


class UserUpdate(BaseModel):
    full_name: str = None
    password: str = None
    is_active: bool = None


@router.get("/users", response_model=List[User])
async def read_users(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="没有权限访问此资源"
        )
    db = Database()
    return db.get_all_users()


@router.get("/users/{username}", response_model=User)
async def read_user(
    username: str, current_user: User = Depends(get_current_active_user)
):
    if current_user.role != "admin" and current_user.username != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="没有权限访问此资源"
        )
    db = Database()
    user = db.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return user


@router.put("/users/{username}")
async def update_user(
    username: str,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
):
    if current_user.role != "admin" and current_user.username != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="没有权限修改此用户"
        )

    db = Database()
    user = db.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        from app.api.endpoints.auth import get_password_hash

        update_data["password"] = get_password_hash(update_data["password"])

    db.update_user(user["_id"], **update_data)
    return {"message": "用户更新成功"}


@router.delete("/users/{username}")
async def delete_user(
    username: str, current_user: User = Depends(get_current_active_user)
):
    if current_user.role != "admin" and current_user.username != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="没有权限删除此用户"
        )

    db = Database()
    user = db.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

    db.delete_user(user["_id"])
    return {"message": "用户删除成功"}
