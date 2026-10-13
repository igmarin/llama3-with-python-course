"""Lesson 8.1: 8-bit vs 4-bit, the same model at two sizes.

Goal:     Measure what quantization costs and saves, on your own machine.
Concept:  The 3B model was trained with 16-bit weights. Quantization stores each
          weight in fewer bits: Q8_0 keeps 8, Q4_K_M about 4.5. Fewer bits mean a
          smaller file, less memory and usually faster generation, because each new
          token requires reading every weight from memory. The cost is some lost
          precision, which may or may not show up in your task's accuracy.
          Each profile runs in its own process, so memory numbers don't mix.
Requires: python scripts/download_model.py --profile laptop-16gb-q8   (3.4 GB)
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/08_quantization/01_compare_quantization.py [--save]
Expected: A table with file size, load time, speed, peak memory and challenge-set
          accuracy for the 4-bit and the 8-bit file. Takes about 2 minutes.
          ``--save`` writes results_<machine>.md next to this file.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from llama_course.benchmark import peak_memory_mb, run_benchmark, summarize
from llama_course.classify import evaluate
from llama_course.data import load_messages
from llama_course.llm import get_llm, get_profile

PROFILES = ("laptop-16gb", "laptop-16gb-q8")
PROMPTS = [
    "Explain in three sentences how a bicycle derailleur works.",
    "Write a short reply to a customer whose bike delivery is late.",
    "List four things to check on a bicycle before a long ride.",
]


def measure(profile_name: str) -> dict:
    """Load one profile and measure it. Runs inside its own process."""
    profile = get_profile(profile_name)
    start = time.perf_counter()
    llm = get_llm(profile_name)
    load_s = time.perf_counter() - start
    speed = summarize(run_benchmark(llm, PROMPTS, runs=1, max_tokens=128))["tokens_per_s"]
    accuracy = evaluate(llm, load_messages("challenge"))["accuracy"]
    return {
        "profile": profile_name,
        "file": profile.file,
        "file_gb": profile.path.stat().st_size / 1e9,
        "load_s": load_s,
        "tokens_per_s": speed,
        "peak_mb": peak_memory_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "challenge": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--one", help=argparse.SUPPRESS)  # internal: measure one profile, print JSON
    parser.add_argument("--save", action="store_true", help="Write the table to a Markdown file.")
    args = parser.parse_args()

    if args.one:
        print("RESULT " + json.dumps(measure(args.one)))
        return

    for name in PROFILES:
        if not get_profile(name).path.is_file():
            sys.exit(f"Missing model for {name}. Run: python scripts/download_model.py --profile {name}")

    results = []
    for name in PROFILES:
        print(f"Measuring {name}...")
        run = subprocess.run([sys.executable, __file__, "--one", name], capture_output=True, text=True, check=True)
        line = next(l for l in run.stdout.splitlines() if l.startswith("RESULT "))
        results.append(json.loads(line[len("RESULT "):]))

    lines = [
        "| Profile | File | Size | Load | Speed | Peak memory | Challenge accuracy (60) |",
        "|---------|------|------|------|-------|-------------|-------------------------|",
    ]
    for r in results:
        lines.append(
            f"| `{r['profile']}` | {r['file']} | {r['file_gb']:.1f} GB | {r['load_s']:.1f} s | "
            f"{r['tokens_per_s']:.1f} tokens/s | {r['peak_mb']:,.0f} MB | {r['challenge']:.1%} |"
        )
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if args.save:
        out = Path(__file__).parent / "results_laptop-16gb.md"
        out.write_text("# 4-bit vs 8-bit, Llama 3.2 3B\n\n" + text)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
