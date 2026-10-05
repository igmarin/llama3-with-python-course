"""Download a GGUF model from HuggingFace into models/."""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit(
        "Missing dependency. Install with: pip install huggingface-hub\n"
        "Or set --url to download directly with curl/wget."
    )

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="HuggingFace repo id, e.g. bartowski/Llama-3.2-3B-Instruct-GGUF",
    )
    group.add_argument(
        "--url",
        help="Direct download URL for a .gguf file (skips HF hub).",
    )
    parser.add_argument(
        "--file",
        default="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        help="Filename inside the HF repo (ignored with --url).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        help=f"Where to save the model (default: {MODELS_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        dest = out_dir / Path(args.url).name
        print(f"Downloading {args.url} -> {dest}")
        os.system(f"curl -L --progress-bar -o {dest} {args.url}")
    else:
        dest = out_dir / args.file
        print(f"Downloading {args.model}/{args.file} -> {dest}")
        local_path = hf_hub_download(
            repo_id=args.model,
            filename=args.file,
            local_dir=str(out_dir),
        )
        if Path(local_path) != dest:
            os.replace(local_path, dest)

    print(f"Done. Model saved to: {dest}")


if __name__ == "__main__":
    main()