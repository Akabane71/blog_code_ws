import uuid
from sqlalchemy.orm import Session
from app.db import SessionLocal, init_db
from app.models import ImageTask, TaskStatus
from app.queue import dequeue, enqueue
from app.image_tasks import run_pipeline, add_event, update_task

def worker_loop():
    WORKER_ID = f"w-{uuid.uuid4().hex[:8]}"
    init_db()
    print(f"[worker] started: {WORKER_ID}")

    while True:
        msg = dequeue()
        if not msg:
            continue

        task_id = msg["task_id"]
        db: Session = SessionLocal()
        try:
            task = db.get(ImageTask, task_id)
            if not task:
                db.close()
                continue

            # 简单取消检查
            if task.status == TaskStatus.CANCELED.value:
                db.close()
                continue

            try:
                run_pipeline(db, task)
            except Exception as e:
                task.retry_count += 1
                add_event(db, task.task_id, task.current_step, f"Error: {e}", task.progress, level="error")

                if task.retry_count <= task.max_retries and task.status != TaskStatus.CANCELED.value:
                    update_task(db, task, status=TaskStatus.PENDING.value)
                    db.commit()
                    enqueue(task.task_id)  # 重新入队
                else:
                    update_task(db, task, status=TaskStatus.FAILED.value, error=str(e))
                    db.commit()

        finally:
            db.close()

if __name__ == "__main__":
    worker_loop()
