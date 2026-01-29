from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass

class ImageTask(Base):
    __tablename__ = "image_tasks"

    id: int
    prompt: str
    status: str
    created_at: str
    updated_at: str


    