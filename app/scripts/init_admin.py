from app.models.database import Database
from app.api.endpoints.auth import get_password_hash


def init_admin():
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
        print("管理员账户创建成功")
    else:
        print("管理员账户已存在")


if __name__ == "__main__":
    init_admin()
