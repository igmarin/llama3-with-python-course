import sys
import types

import pytest

from llama_course import llm
from llama_course.llm import ModelNotFoundError, get_llm, get_profile


class FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_llama_cpp(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.delenv("LLAMA_PROFILE", raising=False)
    monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)
    llm._load.cache_clear()


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    path = tmp_path / "fake.gguf"
    path.touch()
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(path))
    return path


def test_default_profile_is_laptop():
    assert get_profile().name == "laptop-16gb"


def test_env_selects_profile(monkeypatch):
    monkeypatch.setenv("LLAMA_PROFILE", "studio-128gb")
    assert get_profile().name == "studio-128gb"


def test_unknown_profile_lists_choices():
    with pytest.raises(KeyError, match="laptop-16gb"):
        get_profile("nope")


def test_profiles_have_required_fields():
    default, profiles = llm.load_profiles()
    assert default in profiles
    for p in profiles.values():
        assert p.repo and p.file.endswith(".gguf") and p.n_ctx > 0


def test_model_path_override_is_used(model_file):
    assert get_llm().kwargs["model_path"] == str(model_file)


def test_missing_file_names_download_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_MODEL_PATH", str(tmp_path / "missing.gguf"))
    with pytest.raises(ModelNotFoundError, match="scripts/download_model.py --profile laptop-16gb"):
        get_llm()


def test_get_llm_is_cached(model_file):
    assert get_llm() is get_llm()


def test_overrides_reach_llama(model_file):
    assert get_llm(n_ctx=512).kwargs["n_ctx"] == 512

