import importlib.util
import os
from pathlib import Path

import pytest

from llama_course.llm import get_profile


def pytest_collection_modifyitems(items):
    path = Path(os.environ.get("LLAMA_MODEL_PATH") or get_profile().path)
    reason = None
    if importlib.util.find_spec("llama_cpp") is None:
        reason = "llama_cpp is not installed"
    elif not path.is_file():
        reason = f"model file missing: {path}"
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip)
