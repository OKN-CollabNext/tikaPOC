import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def cache_set(key: str, value: dict, expire: int = 300) -> None:
    redis_client.setex(key, expire, json.dumps(value))

def cache_get(key: str) -> dict:
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None
