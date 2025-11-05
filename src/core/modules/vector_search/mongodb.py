import os
from typing import Any, Optional, Type

import pymongo
from loguru import logger
from pymongo import errors

from ..base import ModuleEngine


class MongoDBVectorEngine(ModuleEngine):

    @classmethod
    def from_builded(
        cls,
        name: str = 'mongo-dev',
        mongodbUri: Optional[str] = None,
        mongoCollection: Optional[str] = None,
        limit: int = 12,
        **kwargs,
    ) -> Type[ModuleEngine]:
        try:
            if mongodbUri is None:
                mongodbUri = "mongodb://{}:{}@{}:{}/?authSource={}".format(
                    os.getenv('MONGO_DB_USER', kwargs.get('db_user', 'root')),
                    os.getenv('MONGO_DB_PASS', kwargs.get('db_pass', 'example')),
                    os.getenv('MONGO_DB_HOST', kwargs.get('db_host', 'localhost')),
                    os.getenv('MONGO_DB_PORT', kwargs.get('db_port', '27017')),
                    os.getenv('MONGO_DB_AUTH', kwargs.get('db_auth', 'admin')),
                )
            cls.client = pymongo.MongoClient(mongodbUri)
            cls.db = cls.client[name]
            cls.collection = cls.db[mongoCollection]
            cls.limit = limit

        except Exception as e:
            raise ValueError(f"MongoDB failed to initialize. Error: {e}") from e

        return cls()

    def execute(self, query_vector: Any, **kwargs):
        try:
            unset_stage = {"$unset": "embedding"}
            project_stage = {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    # "product_specs": 1,
                    "color_options": 1,
                    "current_price": 1,
                    "product_promotion": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_vector,
                    "path": "embedding",
                    "numCandidates": 400,
                    "limit": kwargs.get('limit_embedding', self.limit),
                }
            }

            pipeline = [vector_search_stage, unset_stage, project_stage]
            results = self.collection.aggregate(pipeline)

            return list(results)

        except Exception as e:
            raise ValueError(f"Failed to query embeddings. Error details: {e}") from e

    def is_healthy(self) -> bool:
        try:
            self.client.server_info()
            return True
        except errors.ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB server selection timeout. Error: {e}")
            return False
        except errors.ConnectionFailure as e:
            logger.error(f"MongoDB connection failure. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking MongoDB connection. Error: {e}")
            return False
