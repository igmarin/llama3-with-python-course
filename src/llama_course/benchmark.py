"""Measure how fast a model runs on this machine.

Three numbers matter for local models:

- **load time**: how long until the model is ready (reading GBs from disk).
- **time to first token**: how long the model "reads" your prompt before it starts
  answering. It grows with prompt length.
- **tokens per second**: how fast it writes the answer once it has started.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

from llama_course.generation import stream_chat


@dataclass(frozen=True)
class RunStats:
    """Timing of one generation. ``tokens`` counts streamed pieces, about one per token."""

    first_token_s: float
    total_s: float
    tokens: int

    @property
    def tokens_per_s(self) -> float:
        """Speed after the first token, so prompt reading is not counted."""
        generating = self.total_s - self.first_token_s
        return (self.tokens - 1) / generating if self.tokens > 1 and generating > 0 else 0.0


def measure_stream(
    pieces: Iterable[str], clock: Callable[[], float] = time.perf_counter
) -> RunStats:
    """Consume a stream of text pieces and time it.

    Args:
        pieces: Iterator of generated text, such as :func:`stream_chat`.
        clock: Returns the current time in seconds. Replace it in tests.

    Raises:
        ValueError: If the stream produced nothing.
    """
    start = clock()
    first = None
    tokens = 0
    for _ in pieces:
        tokens += 1
        if first is None:
            first = clock() - start
    if first is None:
        raise ValueError("The model produced no tokens")
    return RunStats(first, clock() - start, tokens)


def summarize(runs: list[RunStats]) -> dict[str, float]:
    """Median of each measurement. The median ignores one slow outlier run."""
    return {
        "first_token_s": statistics.median(r.first_token_s for r in runs),
        "tokens_per_s": statistics.median(r.tokens_per_s for r in runs),
        "total_s": statistics.median(r.total_s for r in runs),
        "tokens": statistics.median(r.tokens for r in runs),
    }


def run_benchmark(
    llm: Any,
    prompts: list[str],
    runs: int = 3,
    max_tokens: int = 128,
    clock: Callable[[], float] = time.perf_counter,
) -> list[RunStats]:
    """Generate for every prompt ``runs`` times, after one untimed warm-up.

    The warm-up matters: the first call is slower while caches fill.
    """
    def one(prompt: str) -> RunStats:
        messages = [{"role": "user", "content": prompt}]
        return measure_stream(
            stream_chat(llm, messages, max_tokens=max_tokens, temperature=0.0), clock
        )

    one(prompts[0])
    return [one(p) for p in prompts for _ in range(runs)]


def peak_memory_mb(maxrss: int, platform: str = sys.platform) -> float:
    """Convert ``resource.getrusage(...).ru_maxrss`` to megabytes.

    macOS reports bytes and Linux reports kilobytes. The number covers this process only
    and can understate GPU memory on Apple Silicon, so treat it as a lower bound.
    """
    return maxrss / (1024 * 1024) if platform == "darwin" else maxrss / 1024


def result_dict(
    profile: str,
    load_s: float,
    runs: list[RunStats],
    peak_mb: float,
    machine: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON-serializable benchmark record."""
    return {
        "profile": profile,
        "machine": machine,
        "load_s": load_s,
        "peak_memory_mb": peak_mb,
        "summary": summarize(runs),
        "runs": [asdict(r) | {"tokens_per_s": r.tokens_per_s} for r in runs],
    }


def format_table(result: dict[str, Any]) -> str:
    """Render a benchmark record as a small text table."""
    s = result["summary"]
    rows = [
        ("Profile", result["profile"]),
        ("Machine", f"{result['machine']['platform']}, {result['machine']['ram_gb']:.0f} GB RAM"),
        ("Load time", f"{result['load_s']:.1f} s"),
        ("Time to first token", f"{s['first_token_s']:.2f} s"),
        ("Generation speed", f"{s['tokens_per_s']:.1f} tokens/s"),
        ("Peak memory (process)", f"{result['peak_memory_mb']:.0f} MB"),
        ("Runs", str(len(result["runs"]))),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)
