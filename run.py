"""
运行脚本，用于启动FastAPI系统
"""

import os
import sys
import argparse
import logging
import time
import webbrowser
import threading
import json
import uuid
import random
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import run_server
from app.core.config import FASTAPI_PORT
from app.models.database import Database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def open_browser(fastapi_port, delay=2):
    """
    在新线程中打开浏览器

    Args:
        fastapi_port: FastAPI 服务器端口
        delay: 延迟开始时间（秒）
    """

    def _open_browser():
        time.sleep(delay)  # 等待服务器启动

        # 打开 FastAPI 文档页面
        fastapi_url = f"http://localhost:{fastapi_port}/docs"
        logger.info(f"在浏览器中打开 FastAPI 文档: {fastapi_url}")
        webbrowser.open(fastapi_url)

    browser_thread = threading.Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()


def create_test_data(count=20):
    """
    创建测试数据
    """
    db = Database()

    # 确保至少有一个用户存在
    try:
        from app.utils.security import hash_password

        user = db.get_user_by_username("admin")
        if not user:
            user_id = str(uuid.uuid4())
            db.conn.execute(
                """
                INSERT INTO users (_id, username, password, full_name, created_at, is_active, role)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    "admin",
                    hash_password("admin123"),
                    "管理员",
                    datetime.now().isoformat(),
                    1,
                    "admin",
                ),
            )
            db.conn.commit()
            print("已创建管理员用户")
            user_id = user_id
        else:
            user_id = user["_id"]
    except Exception as e:
        print(f"创建用户失败: {e}")
        user_id = "1"  # 默认用户ID

    # 创建预测记录
    predictions = ["正常行驶", "交通拥堵", "交通事故", "车辆起火"]

    for i in range(count):
        prediction_id = str(uuid.uuid4())
        prediction_type = random.choice(predictions)
        confidence = random.uniform(0.5, 0.99)

        # 创建一个随机的过去日期
        days_ago = random.randint(0, 30)
        created_at = (datetime.now() - timedelta(days=days_ago)).isoformat()

        # 准备预测数据
        prediction_data = {
            "prediction": prediction_type,
            "confidence": confidence,
            "class_name": prediction_type,
            "class_id": predictions.index(prediction_type),
            "all_probabilities": {},
            "model_used": "efficientnet",
            "processing_time_ms": random.uniform(100, 500),
        }

        try:
            db.conn.execute(
                """
                INSERT INTO predictions (
                    _id, user_id, prediction, confidence, 
                    image_path, image_url, username, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    user_id,
                    json.dumps(prediction_data),
                    confidence,
                    "/uploads/test.jpg",
                    "/api/v1/uploads/test.jpg",
                    "admin",
                    created_at,
                ),
            )
            db.conn.commit()
            print(f"已添加测试记录 {i+1}/{count}: {prediction_type}")
        except Exception as e:
            print(f"添加记录失败: {e}")

    print(f"完成添加 {count} 条测试数据")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动交通实况识别系统")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    # 显示欢迎信息
    print("\n" + "=" * 80)
    print("交通实况识别系统 - 启动向导".center(80))
    print("=" * 80)
    print("\n欢迎使用交通实况识别系统！")
    print("\n系统将启动以下服务:")
    print(f"FastAPI 服务器 (端口: {FASTAPI_PORT}) - 提供API和文档")
    print("\n按Ctrl+C可随时停止服务")
    print("=" * 80 + "\n")

    try:
        # 打开浏览器（如果需要）
        if not args.no_browser:
            open_browser(FASTAPI_PORT)

        # 运行应用
        run_server()

    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭服务...")
    except Exception as e:
        logger.error(f"启动系统时出错: {e}")
    finally:
        print("\n" + "=" * 80)
        print("系统已关闭".center(80))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test_data":
        count = 20
        if len(sys.argv) > 2:
            try:
                count = int(sys.argv[2])
            except:
                pass
        create_test_data(count)
    else:
        main()
