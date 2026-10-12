"""Load a Llama model once, from a named profile in ``configs/models.yaml``.

Lessons call :func:`get_llm` instead of building ``Llama(...)`` themselves.
Switch models without touching code::

    export LLAMA_PROFILE=studio-128gb        # pick another profile
    export LLAMA_MODEL_PATH=/path/to/x.gguf  # or point at any GGUF file
"""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelProfile:
    """One entry of ``configs/models.yaml``."""

    name: str
    repo: str
    file: str
    n_ctx: int
    n_gpu_layers: int
    n_threads: int | None = None
    mlx_repo: str | None = None  # same model in MLX format, for fine-tuning on Apple Silicon

    @property
    def path(self) -> Path:
        """Where the downloaded GGUF file is expected to live."""
        return MODELS_DIR / self.file


class ModelNotFoundError(FileNotFoundError):
    """The model file for the selected profile is not on disk."""


def load_profiles(config_path: Path = CONFIG_PATH) -> tuple[str, dict[str, ModelProfile]]:
    """Read the config file.

    Args:
        config_path: YAML file with a ``default`` name and a ``profiles`` mapping.

    Returns:
        The default profile name and all profiles keyed by name.
    """
    data = yaml.safe_load(config_path.read_text())
    profiles = {n: ModelProfile(name=n, **p) for n, p in data["profiles"].items()}
    return data["default"], profiles


def get_profile(name: str | None = None) -> ModelProfile:
    """Return a profile by name, falling back to ``$LLAMA_PROFILE`` then the config default.

    Raises:
        KeyError: If the name is not defined; the message lists valid names.
    """
    default, profiles = load_profiles()
    name = name or os.environ.get("LLAMA_PROFILE") or default
    if name not in profiles:
        raise KeyError(f"Unknown profile {name!r}. Available: {', '.join(profiles)}")
    return profiles[name]


def get_llm(profile: str | None = None, **overrides: Any) -> Llama:
    """Return a cached ``Llama`` instance; the model loads once per process.

    Args:
        profile: Profile name. Defaults to ``$LLAMA_PROFILE`` or the config default.
        **overrides: Extra ``Llama`` keyword arguments (for example ``n_ctx=2048``).

    Raises:
        ModelNotFoundError: If the model file is missing. The message gives the
            download command.
    """
    prof = get_profile(profile)
    path = Path(os.environ.get("LLAMA_MODEL_PATH") or prof.path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model file not found: {path}\n"
            f"Download it with: python scripts/download_model.py --profile {prof.name}"
        )
    return _load(prof, str(path), tuple(sorted(overrides.items())))


@cache
def _load(prof: ModelProfile, path: str, overrides: tuple[tuple[str, Any], ...]) -> Llama:
    from llama_cpp import Llama

    kwargs: dict[str, Any] = {
        "n_ctx": prof.n_ctx,
        "n_gpu_layers": prof.n_gpu_layers,
        "verbose": False,
    }
    if prof.n_threads is not None:
        kwargs["n_threads"] = prof.n_threads
    kwargs.update(overrides)
    llm = Llama(model_path=path, **kwargs)
    # Free Metal resources before interpreter exit; otherwise ggml aborts (exit code 134).
    atexit.register(llm.close)
    return llm
