"""Benchmark a model profile: load time, time to first token, tokens per second.

Examples:
    python scripts/benchmark.py --profile laptop-16gb
    python scripts/benchmark.py --profile studio-128gb --runs 5

Results are saved to benchmarks/<profile>.json so machines can be compared.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time

from llama_course.benchmark import (
    format_table,
    peak_memory_mb,
    result_dict,
    run_benchmark,
)
from llama_course.llm import ROOT, ModelNotFoundError, get_llm, get_profile

PROMPTS = [
    "Explain what a hash table is in three sentences.",
    "Write a short email asking a colleague to review a pull request.",
    "List four differences between a list and a tuple in Python.",
]


def machine_info() -> dict:
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    return {"platform": f"{platform.system()} {platform.machine()}", "ram_gb": ram}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile from configs/models.yaml (default: $LLAMA_PROFILE).")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per prompt.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write a JSON file.")
    args = parser.parse_args()

    try:
        profile = get_profile(args.profile)
    except KeyError as exc:
        sys.exit(exc.args[0])
    print(f"Loading {profile.name} ({profile.file})...")
    start = time.perf_counter()
    try:
        llm = get_llm(profile.name)
    except ModelNotFoundError as exc:
        sys.exit(str(exc))
    except (ValueError, RuntimeError, MemoryError) as exc:  # llama.cpp raises ValueError when loading fails
        sys.exit(
            f"Could not load {profile.name}: {exc}\n"
            "Most often the model is too big for this machine's memory. Try a smaller profile."
        )
    load_s = time.perf_counter() - start

    print(f"Generating ({len(PROMPTS)} prompts x {args.runs} runs, 1 warm-up)...")
    runs = run_benchmark(llm, PROMPTS, runs=args.runs, max_tokens=args.max_tokens)
    peak = peak_memory_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = result_dict(profile.name, load_s, runs, peak, machine_info())
    print("\n" + format_table(result))

    if not args.no_save:
        out = ROOT / "benchmarks" / f"{profile.name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
