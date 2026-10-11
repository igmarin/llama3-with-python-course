"""Download a GGUF model into models/.

Examples:
    python scripts/download_model.py --profile laptop-16gb
    python scripts/download_model.py --repo bartowski/Llama-3.2-3B-Instruct-GGUF \
        --file Llama-3.2-3B-Instruct-Q4_K_M.gguf
"""

import argparse
import sys

from huggingface_hub import hf_hub_download

from llama_course.llm import MODELS_DIR, get_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile name from configs/models.yaml.")
    parser.add_argument("--repo", help="Hugging Face repo id (with --file).")
    parser.add_argument("--file", help="Filename inside the repo (with --repo).")
    args = parser.parse_args()

    if args.profile:
        profile = get_profile(args.profile)
        repo, file = profile.repo, profile.file
    elif args.repo and args.file:
        repo, file = args.repo, args.file
    else:
        sys.exit("Pass --profile NAME, or both --repo and --file.")

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Downloading {repo}/{file} -> {MODELS_DIR}")
    path = hf_hub_download(repo_id=repo, filename=file, local_dir=MODELS_DIR)
    print(f"Done. Model saved to: {path}")


if __name__ == "__main__":
    main()
