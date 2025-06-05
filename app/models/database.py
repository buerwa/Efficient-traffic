"""
数据库连接和操作
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import uuid
import random

from app.core.config import BASE_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库文件路径
DATABASE_PATH = os.path.join(BASE_DIR, "traffic_monitor.db")


class Database:
    """
    数据库管理类，处理SQLite数据库的连接和操作
    用于交通实况识别系统的数据存储
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化数据库连接"""
        try:
            database_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "traffic_monitor.db",
            )
            self.conn = sqlite3.connect(database_path)
            self.conn.row_factory = sqlite3.Row
            # 初始化表
            self._init_tables()
        except sqlite3.Error as e:
            logging.error(f"数据库连接失败: {str(e)}")
            raise

    def _init_tables(self):
        """初始化数据库表"""
        try:
            # 创建用户表
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    _id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    full_name TEXT,
                    role TEXT DEFAULT 'user',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT,
                    avatar_url TEXT
                )
                """
            )

            # 创建预测表
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    _id TEXT PRIMARY KEY,
                    user_id TEXT,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    image_url TEXT,
                    username TEXT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    labels TEXT,
                    model_name TEXT,
                    created_at TEXT,
                    location TEXT,
                    weather TEXT,
                    filename TEXT DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES users (_id)
                )
                """
            )

            # 创建地点表
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS locations (
                    _id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TEXT
                )
                """
            )

            # 创建分析事件表
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics (
                    _id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    data TEXT,
                    user_id TEXT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # 检查表是否包含所需的列
            # 预测表必需的列，为了兼容性考虑
            required_columns = [
                "_id",
                "user_id",
                "prediction",
                "confidence",
                "image_path",
                "image_url",
                "username",
                "timestamp",
                "labels",
                "model_name",
                "created_at",
                "filename",
                "location",
                "weather",
            ]

            # 获取表的当前列结构
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(predictions)")
            existing_columns = {column[1] for column in cursor.fetchall()}

            # 检查是否缺少必需的列
            for col in required_columns:
                if col not in existing_columns:
                    # 添加缺少的列
                    try:
                        default_value = "NULL"
                        if col == "timestamp":
                            default_value = "CURRENT_TIMESTAMP"
                        elif col == "filename":
                            default_value = "''"
                        cursor.execute(
                            f"ALTER TABLE predictions ADD COLUMN {col} TEXT DEFAULT {default_value}"
                        )
                        logger.info(f"为predictions表添加列: {col}")
                    except Exception as e:
                        logger.error(f"添加列时出错: {str(e)}")

            # 同样检查users表必需的列
            required_user_columns = [
                "_id",
                "username",
                "password",
                "full_name",
                "role",
                "is_active",
                "created_at",
                "avatar_url",
            ]

            cursor.execute("PRAGMA table_info(users)")
            existing_user_columns = {column[1] for column in cursor.fetchall()}

            for col in required_user_columns:
                if col not in existing_user_columns:
                    try:
                        default_value = "NULL"
                        if col == "is_active":
                            default_value = "1"
                        elif col == "role":
                            default_value = "'user'"
                        cursor.execute(
                            f"ALTER TABLE users ADD COLUMN {col} TEXT DEFAULT {default_value}"
                        )
                        logger.info(f"为users表添加列: {col}")
                    except Exception as e:
                        logger.error(f"添加users表列时出错: {str(e)}")

            # 检查所有表是否存在，如果不存在则创建
            for table_name in ["locations", "analytics"]:
                cursor.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                )
                if cursor.fetchone() is None:
                    if table_name == "locations":
                        self.conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS locations (
                                _id TEXT PRIMARY KEY,
                                name TEXT UNIQUE NOT NULL,
                                description TEXT,
                                created_at TEXT
                            )
                            """
                        )
                        logger.info("创建locations表")
                    elif table_name == "analytics":
                        self.conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS analytics (
                                _id TEXT PRIMARY KEY,
                                event_type TEXT NOT NULL,
                                data TEXT,
                                user_id TEXT,
                                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                            )
                            """
                        )
                        logger.info("创建analytics表")

            self.conn.commit()
            logger.info("数据库表初始化成功")
        except Exception as e:
            logger.error(f"初始化数据库表失败: {str(e)}")
            raise

    def save_prediction(
        self,
        prediction_data: Dict[str, Any],
        user_id: Optional[str] = None,
        location: Optional[str] = None,
        weather: Optional[str] = None,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        username: Optional[str] = None,
    ) -> str:
        """
        保存预测结果到数据库

        Args:
            prediction_data: 预测结果数据
            user_id: 用户ID（可选）
            location: 地点（可选）
            weather: 天气状况（可选）
            image_path: 图片路径（可选）
            image_url: 图片URL（可选）
            username: 用户名（可选）

        Returns:
            str: 保存的记录ID
        """
        try:
            record_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()

            # 从图片路径中提取文件名
            filename = os.path.basename(image_path) if image_path else "unknown.jpg"

            # 准备标签
            labels = json.dumps(
                ["traffic", prediction_data.get("prediction", "accident")]
            )

            # 确保有模型名称
            model_name = prediction_data.get("model_name", "efficientnet")

            # 确保有时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (
                    _id, prediction, confidence, created_at, user_id, 
                    location, weather, image_path, image_url, username,
                    filename, labels, model_name, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    json.dumps(prediction_data),
                    prediction_data.get("confidence", 0.0),
                    created_at,
                    user_id,
                    location,
                    weather,
                    image_path,
                    image_url,
                    username,
                    filename,
                    labels,
                    model_name,
                    timestamp,
                ),
            )
            self.conn.commit()

            logger.info(f"预测结果已保存，ID: {record_id}")
            return record_id

        except Exception as e:
            logger.error(f"保存预测结果时出错: {e}")
            return None

    def get_predictions(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        skip: int = 0,
        location: Optional[str] = None,
        prediction_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取预测历史记录

        Args:
            user_id: 用户ID（可选，如果提供则只返回该用户的记录）
            limit: 返回记录的最大数量
            skip: 跳过的记录数量（用于分页）
            location: 根据地点筛选（可选）
            prediction_type: 根据预测类型筛选（可选）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）

        Returns:
            List[Dict]: 预测记录列表
        """
        try:
            logger.info(
                f"查询预测记录条件: user_id={user_id}, limit={limit}, skip={skip}, prediction_type={prediction_type}"
            )

            cursor = self.conn.cursor()
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if location:
                query += " AND location = ?"
                params.append(location)
            if prediction_type:
                query += " AND prediction LIKE ?"
                params.append(f"%{prediction_type}%")
            if start_date:
                query += " AND date(created_at) >= date(?)"
                params.append(start_date)
            if end_date:
                query += " AND date(created_at) <= date(?)"
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])

            logger.info(f"执行SQL: {query}")
            logger.info(f"参数: {params}")

            cursor.execute(query, params)
            results = cursor.fetchall()

            logger.info(f"数据库返回 {len(results)} 条记录")

            processed_results = []

            for result in results:
                # 将行转换为字典
                row_dict = dict(result)

                # 将JSON字符串转换回字典
                try:
                    if isinstance(row_dict.get("prediction"), str):
                        prediction_data = json.loads(row_dict["prediction"])
                        # 两种可能的数据结构处理
                        if isinstance(prediction_data, dict):
                            if "prediction" in prediction_data:
                                # 提取预测类型作为主要预测结果
                                row_dict["prediction"] = prediction_data["prediction"]
                            elif "class_name" in prediction_data:
                                # 或者使用class_name作为预测结果
                                row_dict["prediction"] = prediction_data["class_name"]
                            # 保存置信度
                            if "confidence" in prediction_data:
                                row_dict["confidence"] = prediction_data["confidence"]
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"解析JSON预测数据失败: {e}")
                    # 如果无法解析，使用原始字符串

                # 确保图像URL正确
                if row_dict.get("image_url") and not row_dict["image_url"].startswith(
                    ("http", "/")
                ):
                    row_dict["image_url"] = f"/{row_dict['image_url']}"

                # 确保有_id字段
                if "_id" not in row_dict and "id" in row_dict:
                    row_dict["_id"] = row_dict["id"]

                processed_results.append(row_dict)

            if processed_results:
                logger.info(f"处理后的第一条记录: {processed_results[0]}")

            return processed_results

        except Exception as e:
            logger.error(f"获取预测历史记录时出错: {e}")
            return []

    def count_predictions(
        self,
        user_id: Optional[str] = None,
        prediction_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        计算符合条件的预测记录数量

        Args:
            user_id: 用户ID（可选，如果提供则只计算该用户的记录）
            prediction_type: 根据预测类型筛选（可选）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）

        Returns:
            int: 符合条件的记录数量
        """
        try:
            cursor = self.conn.cursor()
            query = "SELECT COUNT(*) as count FROM predictions WHERE 1=1"
            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if prediction_type:
                query += " AND prediction LIKE ?"
                params.append(f"%{prediction_type}%")
            if start_date:
                query += " AND date(created_at) >= date(?)"
                params.append(start_date)
            if end_date:
                query += " AND date(created_at) <= date(?)"
                params.append(end_date)

            cursor.execute(query, params)
            result = cursor.fetchone()
            return result["count"] if result else 0

        except Exception as e:
            logger.error(f"计算预测历史记录数量时出错: {e}")
            return 0

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str = None,
        role: str = "user",
        avatar_url: str = None,
    ):
        """
        创建新用户

        Args:
            username: 用户名
            password: 密码（已加密）
            full_name: 全名
            role: 角色 (user, admin)
            avatar_url: 头像URL

        Returns:
            dict: 新创建的用户信息
        """
        try:
            _id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()

            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (
                    _id, username, password, full_name,
                    created_at, is_active, role, avatar_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _id,
                    username,
                    password,
                    full_name,
                    created_at,
                    1,  # is_active
                    role,
                    avatar_url,
                ),
            )
            self.conn.commit()

            return {
                "_id": _id,
                "username": username,
                "full_name": full_name,
                "created_at": created_at,
                "is_active": True,
                "role": role,
                "avatar_url": avatar_url,
            }
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError("用户名已存在")
            raise e

    def get_user_by_username(self, username):
        """
        根据用户名获取用户

        Args:
            username: 用户名

        Returns:
            dict or None: 用户信息或None（如果不存在）
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT _id, username, password, full_name, created_at, is_active, role, avatar_url
            FROM users
            WHERE username = ?
            """,
            (username,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "_id": row[0],
            "username": row[1],
            "password": row[2],
            "full_name": row[3],
            "created_at": row[4],
            "is_active": bool(row[5]),
            "role": row[6],
            "avatar_url": row[7],
        }

    def update_user(self, user_id, **kwargs):
        """
        更新用户信息

        Args:
            user_id: 用户ID
            **kwargs: 需要更新的字段和值

        Returns:
            bool: 更新是否成功
        """
        valid_fields = {
            "username": "username",
            "password": "password",
            "full_name": "full_name",
            "is_active": "is_active",
            "role": "role",
            "avatar_url": "avatar_url",
        }

        # 构建更新字段
        update_fields = []
        values = []

        for key, value in kwargs.items():
            if key in valid_fields:
                update_fields.append(f"{valid_fields[key]} = ?")
                values.append(value)

        if not update_fields:
            return False

        # 添加用户ID
        values.append(user_id)

        # 执行更新
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            UPDATE users
            SET {', '.join(update_fields)}
            WHERE _id = ?
            """,
            tuple(values),
        )
        self.conn.commit()

        return cursor.rowcount > 0

    def add_analytics_event(
        self, event_type: str, data: Dict[str, Any], user_id: Optional[str] = None
    ) -> None:
        """
        记录分析事件

        Args:
            event_type: 事件类型
            data: 事件数据
            user_id: 用户ID（可选）
        """
        try:
            event_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO analytics (_id, event_type, data, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, event_type, json.dumps(data), user_id, timestamp),
            )
            self.conn.commit()

            logger.debug(f"已记录分析事件: {event_type}")

        except Exception as e:
            logger.error(f"记录分析事件时出错: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            Dict: 包含各种统计信息的字典
        """
        try:
            cursor = self.conn.cursor()

            # 总预测数
            cursor.execute("SELECT COUNT(*) as count FROM predictions")
            total_predictions = cursor.fetchone()["count"]

            # 总用户数
            cursor.execute("SELECT COUNT(*) as count FROM users")
            total_users = cursor.fetchone()["count"]

            stats = {
                "total_predictions": total_predictions,
                "total_users": total_users,
                "class_distribution": {},
                "model_usage": {},
            }

            # 类别分布（这里需要解析JSON查询，SQLite不如MongoDB灵活）
            cursor.execute("SELECT prediction FROM predictions")
            predictions = cursor.fetchall()
            class_counts = {}
            model_counts = {}

            for pred in predictions:
                pred_data = json.loads(pred["prediction"])
                class_name = pred_data.get("class_name")
                model_used = pred_data.get("model_used")

                if class_name:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if model_used:
                    model_counts[model_used] = model_counts.get(model_used, 0) + 1

            stats["class_distribution"] = class_counts
            stats["model_usage"] = model_counts

            return stats

        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")
            return {"error": str(e), "total_predictions": 0, "total_users": 0}

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取特定用户的统计信息

        Args:
            user_id: 用户ID

        Returns:
            Dict: 用户统计信息
        """
        try:
            cursor = self.conn.cursor()

            # 获取30天前的时间戳
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()

            # 总预测数
            cursor.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE user_id = ?",
                (user_id,),
            )
            total_predictions = cursor.fetchone()["count"]

            # 最近30天的预测数
            cursor.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE user_id = ? AND created_at >= ?",
                (user_id, thirty_days_ago),
            )
            recent_predictions = cursor.fetchone()["count"]

            # 初始化结果
            stats = {
                "total_predictions": total_predictions,
                "recent_predictions": recent_predictions,
                "favorite_location": None,
                "most_detected_class": None,
                "class_distribution": {},
                "prediction_by_day": {},
            }

            # 如果没有记录，直接返回
            if total_predictions == 0:
                return stats

            # 获取用户的预测记录
            cursor.execute(
                "SELECT prediction, location, created_at FROM predictions WHERE user_id = ?",
                (user_id,),
            )
            predictions = cursor.fetchall()

            # 分析数据
            class_counts = {}
            location_counts = {}
            day_counts = {}

            for pred in predictions:
                # 解析JSON数据
                pred_data = json.loads(pred["prediction"])
                class_name = pred_data.get("class_name")

                # 统计类别分布
                if class_name:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # 统计地点分布
                location = pred["location"]
                if location:
                    location_counts[location] = location_counts.get(location, 0) + 1

                # 按天统计
                created_date = pred["created_at"].split("T")[0]  # 提取日期部分
                day_counts[created_date] = day_counts.get(created_date, 0) + 1

            # 获取最常用的地点和最常检测到的类别
            if location_counts:
                stats["favorite_location"] = max(
                    location_counts.items(), key=lambda x: x[1]
                )[0]

            if class_counts:
                stats["most_detected_class"] = max(
                    class_counts.items(), key=lambda x: x[1]
                )[0]
                stats["class_distribution"] = class_counts

            stats["prediction_by_day"] = day_counts

            return stats

        except Exception as e:
            logger.error(f"获取用户统计信息时出错: {e}")
            return {"error": str(e), "total_predictions": 0, "recent_predictions": 0}

    def get_locations(self) -> List[Dict[str, Any]]:
        """
        获取预设地点列表

        Returns:
            List[Dict]: 包含地点信息的列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM locations ORDER BY name")
            locations = cursor.fetchall()
            return locations
        except Exception as e:
            logger.error(f"获取地点列表时出错: {e}")
            return []

    def add_location(self, location_data: Dict[str, Any]) -> Optional[str]:
        """
        添加新地点

        Args:
            location_data: 地点数据，包含name和description

        Returns:
            str: 地点ID或None（如果添加失败）
        """
        try:
            location_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()

            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO locations (_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (
                    location_id,
                    location_data.get("name"),
                    location_data.get("description"),
                    created_at,
                ),
            )
            self.conn.commit()

            return location_id
        except sqlite3.IntegrityError:
            logger.error("地点名称已存在")
            return None
        except Exception as e:
            logger.error(f"添加地点时出错: {e}")
            return None

    def generate_mock_prediction_data(
        self, count: int = 10, days_range: int = 30, user_id: Optional[str] = None
    ) -> int:
        """
        生成模拟的交通实况历史数据

        Args:
            count: 要生成的记录数量
            days_range: 天数范围（过去多少天内）
            user_id: 用户ID（可选）

        Returns:
            int: 生成的记录数量
        """
        try:
            # 交通状况类别
            traffic_classes = [
                "畅通",
                "轻度拥堵",
                "中度拥堵",
                "严重拥堵",
                "事故",
                "施工",
                "封路",
                "限行",
            ]

            # 可能的地点
            locations = [
                "北京路口",
                "上海高架",
                "广州大道",
                "深圳湾",
                "杭州西湖",
                "成都环线",
            ]

            # 天气状况
            weather_conditions = ["晴朗", "多云", "小雨", "大雨", "雾", "雪"]

            # 查询生成用户名
            username = None
            if user_id:
                user_info = self.get_user_by_id(user_id)
                if user_info:
                    username = user_info.get("username")

            now = datetime.now()
            generated_count = 0

            for _ in range(count):
                # 随机生成日期（在过去days_range天内）
                random_days = random.randint(0, days_range)
                random_hours = random.randint(0, 23)
                random_minutes = random.randint(0, 59)

                timestamp = now - timedelta(
                    days=random_days, hours=random_hours, minutes=random_minutes
                )

                # 随机选择交通状况
                traffic_class = random.choice(traffic_classes)
                confidence = round(random.uniform(0.65, 0.99), 2)

                # 模拟预测数据
                prediction_data = {
                    "prediction": traffic_class,
                    "confidence": confidence,
                    "model_used": "efficientnet",
                    "all_probabilities": {
                        class_name: round(random.uniform(0.01, 0.2), 2)
                        for class_name in traffic_classes
                        if class_name != traffic_class
                    },
                }

                # 添加选中类别的概率
                prediction_data["all_probabilities"][traffic_class] = confidence

                # 随机选择地点和天气
                location = random.choice(locations)
                weather = random.choice(weather_conditions)

                # 模拟图片路径和URL
                mock_image_filename = f"mock_image_{str(uuid.uuid4())[:8]}.jpg"
                mock_image_path = os.path.join(
                    BASE_DIR, "uploads", "mock", mock_image_filename
                )
                mock_image_url = f"/uploads/mock/{mock_image_filename}"

                # 保存预测结果
                self.save_prediction(
                    prediction_data=prediction_data,
                    user_id=user_id,
                    location=location,
                    weather=weather,
                    image_path=mock_image_path,
                    image_url=mock_image_url,
                    username=username,
                )

                generated_count += 1

            return generated_count

        except Exception as e:
            logger.error(f"生成模拟数据时出错: {e}")
            return 0

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """根据ID获取用户"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE _id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_users(self, search_term: Optional[str] = None) -> List[Dict]:
        """获取所有用户

        Args:
            search_term: 搜索关键词，可按用户名或全名搜索

        Returns:
            List[Dict]: 用户列表
        """
        try:
            cursor = self.conn.cursor()
            if search_term:
                cursor.execute(
                    """
                    SELECT * FROM users 
                    WHERE username LIKE ? OR full_name LIKE ?
                    ORDER BY created_at DESC
                    """,
                    (f"%{search_term}%", f"%{search_term}%"),
                )
            else:
                cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"获取用户列表失败: {str(e)}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM users WHERE _id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_prediction(self, prediction_id: str) -> bool:
        """
        删除预测记录

        Args:
            prediction_id: 预测记录ID

        Returns:
            bool: 删除是否成功
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE _id = ?", (prediction_id,))
            self.conn.commit()
            success = cursor.rowcount > 0
            logger.info(
                f"删除预测记录 {prediction_id}: {'成功' if success else '失败'}"
            )
            return success
        except Exception as e:
            logger.error(f"删除预测记录时出错: {e}")
            return False

    def add_prediction(
        self,
        user_id,
        prediction,
        confidence,
        image_path,
        username=None,
        timestamp=None,
        labels=None,
        model_name=None,
        filename=None,
        **kwargs,
    ):
        """添加新的预测记录

        Args:
            user_id: 用户ID
            prediction: 预测结果（类别）
            confidence: 置信度
            image_path: 图像路径
            username: 用户名（可选）
            timestamp: 时间戳（可选，默认为当前时间）
            labels: 标签列表（可选，JSON格式字符串）
            model_name: 模型名称（可选）
            filename: 文件名（可选）
            **kwargs: 其他可选参数

        Returns:
            新添加记录的ID
        """
        try:
            # 生成唯一ID
            prediction_id = str(uuid.uuid4())

            # 如果没有提供时间戳，使用当前时间
            if not timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 如果没有提供文件名，从图像路径中提取
            if not filename and image_path:
                filename = os.path.basename(image_path)
            elif not filename:
                filename = "unknown.jpg"

            # 如果没有提供标签，使用默认值
            if not labels:
                labels = json.dumps(["traffic", prediction])

            # 如果没有提供模型名称，使用默认值
            if not model_name:
                model_name = "efficientnet"

            # 生成图片URL
            image_url = None
            if image_path:
                # 从完整路径中提取相对路径
                if os.path.isabs(image_path):
                    relpath = os.path.relpath(
                        image_path, os.path.join(os.getcwd(), "uploads")
                    )
                    if not relpath.startswith(".."):
                        image_url = f"/uploads/{relpath}"
                # 如果已经是相对路径，直接使用
                elif not image_path.startswith("/"):
                    image_url = f"/uploads/{image_path}"
                else:
                    image_url = image_path

            # 构造SQL插入语句
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions 
                (_id, user_id, prediction, confidence, image_path, image_url, username, timestamp, labels, model_name, filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    user_id,
                    prediction,
                    confidence,
                    image_path,
                    image_url,
                    username,
                    timestamp,
                    labels,
                    model_name,
                    filename,
                ),
            )
            self.conn.commit()

            logger.info(f"保存预测结果ID: {prediction_id}")
            return prediction_id

        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
            raise

    def get_all_predictions(
        self,
        limit=20,
        prediction_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        获取所有预测记录

        Args:
            limit: 限制返回的记录数量
            prediction_type: 根据预测类型筛选（可选）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）

        Returns:
            list: 预测记录列表（仅包含图片、时间、识别结果和置信度）
        """
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT _id, user_id, prediction, confidence, 
                       image_path, image_url, username, timestamp, created_at, 
                       labels, model_name, filename
                FROM predictions
                WHERE 1=1
            """
            params = []

            if prediction_type:
                query += " AND prediction LIKE ?"
                params.append(f"%{prediction_type}%")
            if start_date:
                query += " AND date(created_at) >= date(?)"
                params.append(start_date)
            if end_date:
                query += " AND date(created_at) <= date(?)"
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # 处理结果，确保image_url和timestamp字段有值
            processed_results = []
            for row in results:
                row_dict = dict(row)

                # 确保image_url字段有值
                if (
                    not row_dict.get("image_url")
                    and row_dict.get("filename")
                    and row_dict.get("username")
                ):
                    row_dict["image_url"] = (
                        f"/uploads/{row_dict['username']}/{row_dict['filename']}"
                    )

                # 确保timestamp字段有值
                if not row_dict.get("timestamp") and row_dict.get("created_at"):
                    row_dict["timestamp"] = row_dict["created_at"]
                elif not row_dict.get("timestamp"):
                    # 如果两者都为空，则使用当前时间
                    row_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                processed_results.append(row_dict)

            return processed_results
        except Exception as e:
            logger.error(f"获取所有预测记录失败: {str(e)}")
            return []

    def get_user_predictions(
        self,
        user_id,
        limit=20,
        prediction_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        获取指定用户的预测记录

        Args:
            user_id: 用户ID
            limit: 限制返回的记录数量
            prediction_type: 根据预测类型筛选（可选）
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）

        Returns:
            list: 预测记录列表（仅包含图片、时间、识别结果和置信度）
        """
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT _id, user_id, prediction, confidence, 
                       image_path, image_url, username, timestamp, created_at, 
                       labels, model_name, filename
                FROM predictions
                WHERE user_id = ?
            """
            params = [user_id]

            if prediction_type:
                query += " AND prediction LIKE ?"
                params.append(f"%{prediction_type}%")
            if start_date:
                query += " AND date(created_at) >= date(?)"
                params.append(start_date)
            if end_date:
                query += " AND date(created_at) <= date(?)"
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # 处理结果，确保image_url和timestamp字段有值
            processed_results = []
            for row in results:
                row_dict = dict(row)

                # 确保image_url字段有值
                if (
                    not row_dict.get("image_url")
                    and row_dict.get("filename")
                    and row_dict.get("username")
                ):
                    row_dict["image_url"] = (
                        f"/uploads/{row_dict['username']}/{row_dict['filename']}"
                    )

                # 确保timestamp字段有值
                if not row_dict.get("timestamp") and row_dict.get("created_at"):
                    row_dict["timestamp"] = row_dict["created_at"]
                elif not row_dict.get("timestamp"):
                    # 如果两者都为空，则使用当前时间
                    row_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                processed_results.append(row_dict)

            return processed_results
        except Exception as e:
            logger.error(f"获取用户 {user_id} 的预测记录失败: {str(e)}")
            return []

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        获取预测统计分析数据

        Returns:
            Dict: 包含各类统计数据
        """
        try:
            cursor = self.conn.cursor()

            # 计算各种预测类型的数量
            prediction_types = {}
            cursor.execute("SELECT prediction FROM predictions")
            rows = cursor.fetchall()

            for row in rows:
                try:
                    prediction = row["prediction"]
                    if isinstance(prediction, str):
                        prediction_data = json.loads(prediction)
                        prediction_type = prediction_data.get("prediction", "未知")
                    else:
                        prediction_type = prediction.get("prediction", "未知")

                    # 可能需要将英文类型转换为中文
                    type_mappings = {
                        "fire": "车辆起火",
                        "sparse_traffic": "交通顺畅",
                        "accident": "交通事故",
                        "dense_traffic": "交通拥堵",
                    }

                    prediction_type = type_mappings.get(
                        prediction_type, prediction_type
                    )
                    prediction_types[prediction_type] = (
                        prediction_types.get(prediction_type, 0) + 1
                    )
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    logger.error(f"解析预测类型失败: {str(e)}")

            # 按月份统计数量
            cursor.execute(
                """
                SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
                FROM predictions
                GROUP BY month
                ORDER BY month
                """
            )

            monthly_counts = {row["month"]: row["count"] for row in cursor.fetchall()}

            # 按置信度分布
            confidence_ranges = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}

            cursor.execute("SELECT confidence FROM predictions")
            for row in cursor.fetchall():
                confidence = row["confidence"]
                if confidence < 0.25:
                    confidence_ranges["0-25%"] += 1
                elif confidence < 0.5:
                    confidence_ranges["25-50%"] += 1
                elif confidence < 0.75:
                    confidence_ranges["50-75%"] += 1
                else:
                    confidence_ranges["75-100%"] += 1

            # 计算平均置信度
            cursor.execute("SELECT AVG(confidence) as avg_confidence FROM predictions")
            avg_confidence = cursor.fetchone()["avg_confidence"] or 0

            # 返回统计结果
            return {
                "prediction_types": prediction_types,
                "monthly_counts": monthly_counts,
                "confidence_ranges": confidence_ranges,
                "avg_confidence": avg_confidence,
                "total_count": len(rows),
            }

        except Exception as e:
            logger.error(f"获取预测统计分析失败: {str(e)}")
            return {
                "prediction_types": {},
                "monthly_counts": {},
                "confidence_ranges": {},
                "avg_confidence": 0,
                "total_count": 0,
                "error": str(e),
            }

    def get_confidence_trend(self, days=30, count=None):
        """获取置信度趋势数据

        Args:
            days: 获取最近多少天的趋势
            count: 获取最近多少条记录（如果提供，优先使用count）

        Returns:
            List[Dict]: 按日期分组的置信度趋势数据
        """
        try:
            cursor = self.conn.cursor()

            if count is not None:
                # 基于记录数量的查询
                query = """
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    AVG(confidence) as avg_confidence
                FROM predictions
                GROUP BY date
                ORDER BY date DESC
                LIMIT ?
                """
                cursor.execute(query, (count,))
            else:
                # 基于天数的查询
                query = """
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    AVG(confidence) as avg_confidence
                FROM predictions
                WHERE timestamp >= datetime('now', ?, 'localtime')
                GROUP BY date
                ORDER BY date DESC
                """
                cursor.execute(query, (f"-{days} days",))

            result = []
            for row in cursor.fetchall():
                result.append(
                    {"date": row["date"], "avg_confidence": row["avg_confidence"]}
                )

            return {"data": result}
        except Exception as e:
            logger.error(f"获取置信度趋势失败: {str(e)}")
            return {"data": []}

    def rebuild_predictions_table(self):
        """重建预测记录表"""
        try:
            cursor = self.conn.cursor()

            # 先备份当前数据（可选）
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS predictions_backup AS SELECT * FROM predictions"
            )
            self.conn.commit()

            # 删除现有表
            cursor.execute("DROP TABLE IF EXISTS predictions")

            # 重建表结构
            cursor.execute(
                """
            CREATE TABLE predictions (
                _id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT,
                image_url TEXT,
                username TEXT,
                model_name TEXT,
                labels TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (_id)
            )
            """
            )

            self.conn.commit()
            logger.info("预测记录表重建成功")
            return True
        except Exception as e:
            logger.error(f"重建预测记录表失败: {str(e)}")
            self.conn.rollback()
            return False

    def drop_and_recreate_predictions_table(self):
        """删除预测表并重建"""
        try:
            # 首先检查表是否存在
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # 删除预测表
                self.conn.execute("DROP TABLE predictions")
                logger.info("预测表已删除")

            # 重建预测表
            self.conn.execute(
                """
                CREATE TABLE predictions (
                    _id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    image_url TEXT,
                    username TEXT,
                    model_name TEXT,
                    labels TEXT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (_id)
                )
            """
            )

            self.conn.commit()
            logger.info("预测表重建成功")
            return {"success": True, "message": "预测表已成功删除并重建"}
        except Exception as e:
            logger.error(f"删除并重建预测表失败: {str(e)}")
            self.conn.rollback()
            return {"success": False, "message": f"操作失败: {str(e)}"}
