"""Check the built project page for missing assets, anchors, and stale placeholders."""
import argparse
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.references = []
        self.missing_alt = []
        self.duplicates = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        for attr in ("href", "src"):
            if attr in attrs:
                self.references.append(attrs[attr])
        if tag == "img" and not attrs.get("alt"):
            self.missing_alt.append(attrs.get("src"))


def check(out):
    html = (out / "index.html").read_text(encoding="utf-8")
    if "{{" in html or "PROJECT_PAGE_URL" in html:
        raise ValueError("Unresolved website placeholder")
    page = Page()
    page.feed(html)
    if page.duplicates or page.missing_alt:
        raise ValueError(f"Duplicate IDs: {page.duplicates}; images without descriptions: {page.missing_alt}")
    for ref in page.references:
        url = urlsplit(ref)
        if url.scheme or url.netloc:
            continue
        target = (out / unquote(url.path)).resolve() if url.path else out / "index.html"
        if not target.is_relative_to(out.resolve()) or not target.exists():
            raise ValueError(f"Missing or out-of-artifact asset: {ref}")
        if url.fragment and target.name == "index.html" and url.fragment not in page.ids:
            raise ValueError(f"Missing section: {ref}")
    if hashlib.sha256((out / "paper.pdf").read_bytes()).digest() != hashlib.sha256((ROOT / "docs/paper.pdf").read_bytes()).digest():
        raise ValueError("Website PDF differs from the canonical manuscript")
    print(f"Verified {len(page.references)} links/assets, {len(page.ids)} unique IDs, image descriptions, and the current paper PDF.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / ".site")
    args = parser.parse_args()
    check(args.site_dir.resolve())


if __name__ == "__main__":
    main()
