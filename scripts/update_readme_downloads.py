"""Refresh the cumulative HuggingFace download count in README.md.

The count lives between the HF_DOWNLOADS_START / HF_DOWNLOADS_END HTML comment
markers, which render as nothing on GitHub. Run on a daily schedule by
.github/workflows/update-hf-downloads.yml, or by hand from the project root.

Note: the HuggingFace API only returns `downloadsAllTime` when it is requested
via `expand`; the default `downloads` field is a rolling 30-day figure.
"""

import json
import pathlib
import re
import sys
import urllib.request

REPO_ID = "jurinho17-sv/news-article-bias-classifier"
API_URL = f"https://huggingface.co/api/models/{REPO_ID}?expand=downloadsAllTime"
MARKERS = re.compile(
    r"(<!-- HF_DOWNLOADS_START -->)(.*?)(<!-- HF_DOWNLOADS_END -->)", re.DOTALL
)

readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
text = readme.read_text(encoding="utf-8")

match = MARKERS.search(text)
if match is None:
    print("❌ HF_DOWNLOADS markers not found in README.md")
    sys.exit(1)

with urllib.request.urlopen(API_URL, timeout=30) as response:
    downloads = int(json.load(response)["downloadsAllTime"])

if match.group(2) == str(downloads):
    print(f"✅ Already up to date at {downloads} downloads")
    sys.exit(0)

readme.write_text(MARKERS.sub(rf"\g<1>{downloads}\g<3>", text), encoding="utf-8")
print(f"✅ Updated {match.group(2)} -> {downloads} downloads")
