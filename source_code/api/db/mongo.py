import logging
import os
from typing import Optional, Union, Dict, List

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_PORT = int(os.environ.get("MONGO_PORT", 27017))
MONGO_DB = os.environ.get("MONGO_DB", "agent")


class AsyncMongoDB:
    """
    异步 MongoDB 连接类
    基于 Motor 驱动实现，适用于 FastAPI 等异步框架
    """

    def __init__(
            self,
            uri: str = "mongodb://localhost:27017",
            database: Optional[str] = None,
            **kwargs
    ):
        """
        初始化异步 MongoDB 连接

        :param uri: MongoDB 连接URI，例如: mongodb://username:password@host:port
        :param database: 默认数据库名称
        :param kwargs: 其他传递给 AsyncIOMotorClient 的参数
        """
        self.uri = uri
        self.database_name = database
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.connection_kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """
        连接到 MongoDB 服务器

        :return: 是否连接成功
        """
        try:
            self.client = AsyncIOMotorClient(self.uri, **self.connection_kwargs)

            # 测试连接
            await self.client.admin.command('ping')

            if self.database_name:
                self.db = self.client[self.database_name]

            self.logger.info("成功连接到 MongoDB")
            return True
        except Exception as e:
            self.logger.error(f"连接 MongoDB 失败: {e}")
            return False

    async def close(self):
        """关闭 MongoDB 连接"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.logger.info("已关闭 MongoDB 连接")

    def get_database(self, db_name: str):
        """
        获取数据库对象

        :param db_name: 数据库名称
        :return: 数据库对象
        """
        if not self.client:
            raise RuntimeError("MongoDB 连接未建立")
        return self.client[db_name]

    def get_collection(self, collection_name: str, db_name: Optional[str] = None):
        """
        获取集合对象

        :param collection_name: 集合名称
        :param db_name: 数据库名称 (可选)
        :return: 集合对象
        """
        if db_name:
            db = self.client[db_name]
        elif self.db is not None:
            db = self.db
        else:
            raise ValueError("未指定数据库名称且无默认数据库")

        return db[collection_name]

    # CRUD 操作

    async def insert_one(
            self,
            collection_name: str,
            document: Dict,
            db_name: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        插入单个文档

        :param collection_name: 集合名称
        :param document: 要插入的文档
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他插入选项
        :return: 插入文档的 _id
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.insert_one(document, **kwargs)
        return str(result.inserted_id)

    async def insert_many(
            self,
            collection_name: str,
            documents: List[Dict],
            db_name: Optional[str] = None,
            **kwargs
    ) -> List[str]:
        """
        插入多个文档

        :param collection_name: 集合名称
        :param documents: 文档列表
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他插入选项
        :return: 插入文档的 _id 列表
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.insert_many(documents, **kwargs)
        return [str(_id) for _id in result.inserted_ids]

    async def find_one(
            self,
            collection_name: str,
            query: Optional[Dict] = None,
            db_name: Optional[str] = None,
            **kwargs
    ) -> Optional[Dict]:
        """
        查询单个文档

        :param collection_name: 集合名称
        :param query: 查询条件 (可选)
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他查询选项
        :return: 查询到的文档或 None
        """
        collection = self.get_collection(collection_name, db_name)
        return await collection.find_one(query or {}, **kwargs)

    async def find(
            self,
            collection_name: str,
            query: Optional[Dict] = None,
            db_name: Optional[str] = None,
            **kwargs
    ) -> List[Dict]:
        """
        查询多个文档

        :param collection_name: 集合名称
        :param query: 查询条件 (可选)
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他查询选项
        :return: 文档列表
        """
        collection = self.get_collection(collection_name, db_name)
        cursor = collection.find(query or {}, **kwargs)
        return await cursor.to_list(length=None)

    async def update_one(
            self,
            collection_name: str,
            query: Dict,
            update: Dict,
            db_name: Optional[str] = None,
            **kwargs
    ) -> Dict:
        """
        更新单个文档

        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新操作
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他更新选项
        :return: 更新结果信息
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.update_one(query, update, **kwargs)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None
        }

    async def update_many(
            self,
            collection_name: str,
            query: Dict,
            update: Dict,
            db_name: Optional[str] = None,
            **kwargs
    ) -> Dict:
        """
        更新多个文档

        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新操作
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他更新选项
        :return: 更新结果信息
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.update_many(query, update, **kwargs)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None
        }

    async def delete_one(
            self,
            collection_name: str,
            query: Dict,
            db_name: Optional[str] = None,
            **kwargs
    ) -> int:
        """
        删除单个文档

        :param collection_name: 集合名称
        :param query: 查询条件
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他删除选项
        :return: 删除的文档数量
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.delete_one(query, **kwargs)
        return result.deleted_count

    async def delete_many(
            self,
            collection_name: str,
            query: Dict,
            db_name: Optional[str] = None,
            **kwargs
    ) -> int:
        """
        删除多个文档

        :param collection_name: 集合名称
        :param query: 查询条件
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他删除选项
        :return: 删除的文档数量
        """
        collection = self.get_collection(collection_name, db_name)
        result = await collection.delete_many(query, **kwargs)
        return result.deleted_count

    # 其他操作

    async def count_documents(
            self,
            collection_name: str,
            query: Optional[Dict] = None,
            db_name: Optional[str] = None,
            **kwargs
    ) -> int:
        """
        计算文档数量

        :param collection_name: 集合名称
        :param query: 查询条件 (可选)
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他计数选项
        :return: 文档数量
        """
        collection = self.get_collection(collection_name, db_name)
        return await collection.count_documents(query or {}, **kwargs)

    async def aggregate(
            self,
            collection_name: str,
            pipeline: List[Dict],
            db_name: Optional[str] = None,
            **kwargs
    ) -> List[Dict]:
        """
        聚合操作

        :param collection_name: 集合名称
        :param pipeline: 聚合管道
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他聚合选项
        :return: 聚合结果
        """
        collection = self.get_collection(collection_name, db_name)
        cursor = collection.aggregate(pipeline, **kwargs)
        return await cursor.to_list(length=None)

    async def create_index(
            self,
            collection_name: str,
            keys: Union[str, List[tuple]],
            db_name: Optional[str] = None,
            **kwargs
    ) -> str:
        """
        创建索引

        :param collection_name: 集合名称
        :param keys: 索引键
        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他索引选项
        :return: 创建的索引名称
        """
        collection = self.get_collection(collection_name, db_name)
        return await collection.create_index(keys, **kwargs)

    async def list_collections(
            self,
            db_name: Optional[str] = None,
            **kwargs
    ) -> List[str]:
        """
        列出数据库中的所有集合

        :param db_name: 数据库名称 (可选)
        :param kwargs: 其他选项
        :return: 集合名称列表
        """
        if db_name:
            db = self.client[db_name]
        elif self.db:
            db = self.db
        else:
            raise ValueError("未指定数据库名称且无默认数据库")

        return await db.list_collection_names(**kwargs)

    async def __aenter__(self):
        """异步上下文管理器进入时自动连接"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出时自动关闭连接"""
        await self.close()


# 使用示例
async def example_usage():
    # 创建连接实例
    mongo = AsyncMongoDB(
        uri="mongodb://username:password@localhost:27017",
        database="test_db"
    )

    # 使用异步上下文管理器
    async with mongo:
        # 插入文档
        doc_id = await mongo.insert_one("users", {"name": "Alice", "age": 30})
        print(f"插入文档 ID: {doc_id}")

        # 查询文档
        user = await mongo.find_one("users", {"name": "Alice"})
        print(f"查询到的用户: {user}")

        # 更新文档
        update_result = await mongo.update_one(
            "users",
            {"name": "Alice"},
            {"$set": {"age": 31}}
        )
        print(f"更新结果: {update_result}")

        # 再次查询
        updated_user = await mongo.find_one("users", {"name": "Alice"})
        print(f"更新后的用户: {updated_user}")

        # 删除文档
        deleted_count = await mongo.delete_one("users", {"name": "Alice"})
        print(f"删除的文档数量: {deleted_count}")


async def get_db():
    db = AsyncMongoDB(uri=f"mongodb://{MONGO_HOST}:{MONGO_PORT}", database=MONGO_DB)
    await db.connect()
    return db



# 在 FastAPI 中使用示例
"""
from fastapi import FastAPI, Depends
app = FastAPI()

async def get_db():
    db = AsyncMongoDB(uri="mongodb://localhost:27017", database="myapp")
    await db.connect()
    try:
        yield db
    finally:
        await db.close()

@app.get("/users/{username}")
async def get_user(username: str, db: AsyncMongoDB = Depends(get_db)):
    user = await db.find_one("users", {"username": username})
    return user
"""
