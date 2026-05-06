import json
import os
import shutil
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def copy_artifact(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst) or ".")
    shutil.copyfile(src, dst)