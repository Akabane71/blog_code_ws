from fastapi import FastAPI,APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.db import SessionLocal
from sqlalchemy.orm import Session
from app.models import ImageTask, TaskEvent
import uuid
import json
import time
from app.db import init_db
from app.queue import enqueue
from app.models import ImageTaskParams
app = FastAPI()
router = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
)

class CreateTaskReq(BaseModel):
    params: dict
    idempotency_key: str | None = None


@app.on_event("startup")
def on_startup():
    init_db()

@router.post("/tasks")
def create_task(req: CreateTaskReq):
    db: Session = SessionLocal()
    try:
        idem = req.idempotency_key or ""
        if idem:
            existing = db.query(ImageTask).filter(ImageTask.idempotency_key == idem).first()
            if existing:
                return {"task_id": existing.task_id, "status": existing.status, "deduped": True}

        task_id = uuid.uuid4().hex
        task = ImageTask(
            task_id=task_id,
            status="PENDING",
            progress=0,
            current_step="",
            params_json= ImageTaskParams(
                width=req.params.get("width",512),
                height=req.params.get("height",512),
                prompt=req.params.get("prompt","给我画一个猫娘！"),
            ).model_dump_json(),
            idempotency_key=idem,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        db.add(task)
        db.commit()

        enqueue(task_id)
        return {"task_id": task_id, "status": "PENDING"}
    finally:
        db.close()


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    db: Session = SessionLocal()
    try:
        task = db.get(ImageTask, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "current_step": task.current_step,
            "retry_count": task.retry_count,
            "result": task.result_json,
            "error": task.error,
            "updated_at": task.updated_at,
        }
    finally:
        db.close()


@router.get("/tasks/{task_id}/events")
def get_events(task_id: str, limit: int = 100):
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(TaskEvent)
            .filter(TaskEvent.task_id == task_id)
            .order_by(TaskEvent.id.asc())
            .limit(limit)
            .all()
        )
        return [
            {"id": r.id, "ts": r.ts, "step": r.step, "progress": r.progress, "level": r.level, "message": r.message}
            for r in rows
        ]
    finally:
        db.close()

@router.post("/tasks/{task_id}/cancel")
def cancel(task_id: str):
    db: Session = SessionLocal()
    try:
        task = db.get(ImageTask, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        task.status = "CANCELED"
        task.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        db.commit()
        return {"task_id": task_id, "status": "CANCELED"}
    finally:
        db.close()


app.include_router(router)
