import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
DATA_DIR = Path("data")
VECTORSTORE_DIR = "vectorstores"

# runtime.py
from typing import Optional

ACTIVE_DOC_NAME: Optional[str] = None
ACTIVE_CHAIN = None


def textfile_exists():
    text_files = [p for p in DATA_DIR.iterdir() if p.suffix == ".txt"]
    return text_files

def search_vectorstore(file):
    p = Path(__file__).resolve().parent / "vectorstores"
    for f in p.iterdir():
        if file == f.name:
            return True
    return False

def index_name_from_path(file_path: str) -> str:
    return file_path.stem

def index_path_from_file(file_path: str) -> str:
    return os.path.join(VECTORSTORE_DIR, index_name_from_path(file_path))
