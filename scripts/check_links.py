"""Check that every link in the docs works: web links resolve, relative file links exist.

    python scripts/check_links.py                 # README.md, docs/, lessons/, data/
    python scripts/check_links.py some/file.md    # specific files or folders
    python scripts/check_links.py --offline       # relative links only, no web requests (CI)

Exits with 1 if any link fails. Some sites reject scripts with 403 or 429; the report
marks those "blocked" so you can open them in a browser instead of treating them as dead.
"""

import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = re.compile(r"https?://[^\s<>\"')\]`]+")
HEADERS = {"User-Agent": "Mozilla/5.0 (link checker for llama3-with-python-course)"}


def find_links(text: str) -> set[str]:
    """Return the distinct URLs in ``text``, without trailing punctuation."""
    return {m.rstrip(".,;:") for m in URL.findall(text)}


RELATIVE = re.compile(r"\]\((?!https?://|mailto:|#)([^)#\s]+)")


def broken_relative_links(path: Path) -> list[str]:
    """Return relative Markdown link targets in ``path`` that point to missing files."""
    return [
        target
        for target in RELATIVE.findall(path.read_text())
        if not (path.parent / target).exists()
    ]


def collect(paths: list[Path]) -> dict[str, list[str]]:
    """Map each URL to the files that contain it."""
    links: dict[str, list[str]] = {}
    for path in paths:
        files = [path] if path.is_file() else [*path.rglob("*.md"), *path.rglob("*.py")]
        for f in files:
            if "handoffs" in f.parts or ".venv" in f.parts:
                continue
            for url in find_links(f.read_text()):
                links.setdefault(url, []).append(str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f))
    return links


def check(url: str, attempts: int = 3) -> str:
    """Return "ok", "blocked" (403/429, may be fine in a browser) or an error text.

    Server errors (5xx) and network errors are retried with a pause: busy sites such as
    GitHub answer 503 when many requests arrive at once, and DNS lookups fail now and then.
    """
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=20):
                return "ok"
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return "blocked" if exc.code in (403, 429) else f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < attempts - 1:  # DNS lookups fail now and then under load
                time.sleep(2 * (attempt + 1))
                continue
            return f"error: {exc}"
    return "unreachable"


def main() -> None:
    offline = "--offline" in sys.argv
    args = [Path(a).resolve() for a in sys.argv[1:] if a != "--offline"]
    paths = args or [ROOT / "README.md", ROOT / "docs", ROOT / "lessons", ROOT / "data"]
    links = collect([p for p in paths if p.exists()])
    missing = [
        f"{f.relative_to(ROOT)}: {target}"
        for p in paths if p.exists()
        for f in ([p] if p.is_file() else p.rglob("*.md"))
        if f.suffix == ".md" and "handoffs" not in f.parts and ".venv" not in f.parts and f.is_relative_to(ROOT)
        for target in broken_relative_links(f)
    ]
    for item in missing:
        print(f"missing file   {item}")
    if offline:
        print(f"Checked relative links only; {len(missing)} missing.")
        sys.exit(1 if missing else 0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip(links, pool.map(check, links)))
    bad = {u: r for u, r in results.items() if r != "ok"}
    for url, status in sorted(bad.items()):
        print(f"{status:<14} {url}   ({', '.join(links[url])})")
    print(f"\n{len(links) - len(bad)} of {len(links)} links ok.")
    sys.exit(1 if missing or any(r != "blocked" for r in bad.values()) else 0)


if __name__ == "__main__":
    main()
