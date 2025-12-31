from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    opensearch_host: str = os.getenv("OPENSEARCH_HOST", "https://localhost:9200")
    opensearch_user: str = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD", "admin"))
    index_name: str = os.getenv("OPENSEARCH_INDEX", "local-notes")
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    cache_path: Path = Path(os.getenv("CACHE_PATH", "tmp/index.json"))


settings = Settings()
