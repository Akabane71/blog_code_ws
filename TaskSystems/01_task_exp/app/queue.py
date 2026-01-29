import redis
import json

r = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)

QUEUE_KEY = "task_queue"

def enqueue(task_id: str):
    # 消息只放最小指针
    r.lpush(QUEUE_KEY, json.dumps({"task_id": task_id}))

def dequeue(block_sec: int = 5):
    item = r.brpop(QUEUE_KEY, timeout=block_sec)
    if not item:
        return None
    _, raw = item
    return json.loads(raw)
