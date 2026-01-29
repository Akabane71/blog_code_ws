import time
import json
from sqlalchemy.orm import Session
from app.models import ImageTask, TaskEvent,TaskStatus

def add_event(db: Session, task_id: str, step: str, msg: str, progress: int, level: str = "info"):
    ev = TaskEvent(task_id=task_id, step=step, message=msg, progress=progress, level=level, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    db.add(ev)

def update_task(db: Session, task: ImageTask, status: str | None = None, step: str | None = None, progress: int | None = None, result: dict | None = None, error: str | None = None):
    if status is not None:
        task.status = status
    if step is not None:
        task.current_step = step
    if progress is not None:
        task.progress = progress
    if result is not None:
        task.result_json = json.dumps(result, ensure_ascii=False)
    if error is not None:
        task.error = error
    task.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

def run_pipeline(db: Session, task: ImageTask):
    # Step 1: "init"
    running = TaskStatus.RUNNING.value
    update_task(db, task, status=running, step="init", progress=5)
    add_event(db, task.task_id, "init", "Initializing image task...", 5)
    db.commit()
    time.sleep(2)

    # Step 2: "process"
    process = TaskStatus.RUNNING.value
    update_task(db, task, status=process, step="process", progress=40)
    add_event(db, task.task_id, "process", "Processing data (GPU heavy)...", 40)
    db.commit()
    time.sleep(4)

    # Done
    result = {"ok": True, "summary": "Image processed successfully" ,"img_path":"/static/images/result.png"}
    update_task(db, task, status=TaskStatus.SUCCESS.value, step="done", progress=100, result=result)
    add_event(db, task.task_id, "done", "Task finished.", 100)
    db.commit()
