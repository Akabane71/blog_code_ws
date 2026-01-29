
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Text, DateTime, func
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, time

def now_time() -> str:
    """返回当前时间的格式字符串"""
    return time.strftime("%Y-%m-%dT%H:%M:%S")

class TaskStyle(str, Enum):
    IMG = "IMAGE"
    NORMAL = "NORMAL"

class ImageTaskParams(BaseModel):
    width: int = 512
    height: int = 512
    prompt: str = ""
    
class ImageTaskRequest(BaseModel):
    style: TaskStyle
    params: ImageTaskParams

    
class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class Base(DeclarativeBase):
    pass

class ImageTask(Base):
    __tablename__ = "image_tasks"
    # 任务ID 主键
    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 当前状态
    status: Mapped[str] = mapped_column(String(32), default=TaskStatus.PENDING.value)  # PENDING/RUNNING/SUCCESS/FAILED/CANCELED
    # 进度
    progress: Mapped[int] = mapped_column(Integer, default=0)
    # 当前步骤
    current_step: Mapped[str] = mapped_column(String(64), default="")
    # 任务参数
    params_json: Mapped[str] = mapped_column(Text, default=ImageTaskParams().model_dump_json())

    # 防止提交重复任务
    idempotency_key: Mapped[str] = mapped_column(String(128), default="", index=True)
    # 失败重试的计数
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    # 最大失败重试数
    max_retries: Mapped[int] = mapped_column(Integer, default=2)

    # 分布式事务锁
    locked_by: Mapped[str] = mapped_column(String(64), default="")
    # 锁定到期时间（ISO字符串）
    lock_until: Mapped[str] = mapped_column(String(64), default="")  
    
    # 结果数据
    result_json: Mapped[str] = mapped_column(Text, default="")
    # 错误信息
    error: Mapped[str] = mapped_column(Text, default="")

    # 创建事件时间（自动填充当前时间）
    created_at: Mapped[str] = mapped_column(String(64), default=lambda: now_time)
    # 更新时间（自动填充当前时间）
    updated_at: Mapped[str] = mapped_column(String(64), default=lambda: now_time)


class TaskEvent(Base):
    __tablename__ = "task_events"
    # 主键，自增
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 关联任务ID
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    # 步骤
    step: Mapped[str] = mapped_column(String(64), default="")
    # 消息内容
    message: Mapped[str] = mapped_column(Text, default="")
    # 进度
    progress: Mapped[int] = mapped_column(Integer, default=0)
    # 日志级别
    level: Mapped[str] = mapped_column(String(16), default="info")
    # 时间戳
    ts: Mapped[str] = mapped_column(String(64), default=lambda: "")
