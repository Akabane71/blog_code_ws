from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

engine = create_engine("sqlite:///./tasks.db", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)
