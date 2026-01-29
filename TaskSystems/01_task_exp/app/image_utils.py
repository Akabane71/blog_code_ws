from datetime import datetime
from app.models import ImageTaskParams, ImageTask, TaskStatus,TaskEvent


def generate_task_id() -> str:
    import uuid
    return str(uuid.uuid4())

def create_image_task(img_tsk_params: ImageTaskParams)-> ImageTask:
    return ImageTask(
        task_id = generate_task_id(),
        status = TaskStatus.PENDING.value,
        progress = 0,
        current_step = "",
        idempotency_key = "",
        retry_count = 0,
        max_retries = 2,
        params_json=img_tsk_params.model_dump_json(),
        created_at = "",
        updated_at = "",
    )
    
def create_image_task_envt(img_tsk: ImageTask)->TaskEvent:
    return TaskEvent(
        event_id = generate_task_id(),
        task_id = img_tsk.task_id,
        event_type = "",
        event_description = "",
        created_at = datetime.utcnow(),
    )