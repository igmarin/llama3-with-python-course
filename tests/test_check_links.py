import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_links", Path(__file__).parents[1] / "scripts" / "check_links.py")
check_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_links)


def test_find_links_strips_markdown_and_punctuation():
    text = "See [docs](https://a.example/x/y). Also <https://b.example/z>, and https://c.example/q?x=1."
    assert check_links.find_links(text) == {"https://a.example/x/y", "https://b.example/z", "https://c.example/q?x=1"}


def test_no_links_is_empty():
    assert check_links.find_links("nothing here") == set()


def test_broken_relative_links_are_found(tmp_path):
    (tmp_path / "real.md").write_text("x")
    page = tmp_path / "page.md"
    page.write_text("[ok](real.md) [anchor](real.md#top) [web](https://x.example) [gone](missing.md) [top](#top)")
    assert check_links.broken_relative_links(page) == ["missing.md"]
