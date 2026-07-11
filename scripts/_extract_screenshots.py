"""One-off helper: pull the user's screenshots out of the filled-in draft and
record which report section each was placed under.

Walks the DOCX in document order, tracks the current Heading, and for every
inline image decides whether it is one of our auto-generated figures (matched
by md5 against reports/figures/*.png -> skipped, the generator re-adds those)
or a user screenshot (saved to reports/screenshots/extracted/ and recorded in
a manifest keyed by section heading, in the order they appear).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

from docx import Document
from docx.oxml.ns import qn

SRC = "reports/Heart_Disease_MLOps_Report_DRAFT.docx"
OUT_DIR = "reports/screenshots/extracted"
MANIFEST = "reports/screenshots/manifest.json"

os.makedirs(OUT_DIR, exist_ok=True)

fig_hashes = {
    hashlib.md5(open(f, "rb").read()).hexdigest(): os.path.basename(f)
    for f in glob.glob("reports/figures/*.png")
}

d = Document(SRC)


def para_style(p_el) -> str:
    ppr = p_el.find(qn("w:pPr"))
    if ppr is not None:
        st = ppr.find(qn("w:pStyle"))
        if st is not None:
            return st.get(qn("w:val")) or ""
    return ""


current = "PREAMBLE"
manifest: dict[str, list[str]] = {}
counter = 0

for el in d.element.body.iterchildren():
    if el.tag != qn("w:p"):
        continue
    text = "".join(t.text for t in el.iter(qn("w:t")) if t.text).strip()
    style = para_style(el)
    # Heading detection (styles like 'Heading1')
    if style.startswith("Heading") and text:
        current = text
        manifest.setdefault(current, [])
        continue
    # Images embedded in this paragraph
    for blip in el.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if not rid:
            continue
        part = d.part.related_parts.get(rid)
        if part is None:
            continue
        blob = part.blob
        h = hashlib.md5(blob).hexdigest()
        if h in fig_hashes:
            continue  # our own figure; generator re-adds it
        counter += 1
        ext = ".png" if part.content_type.endswith("png") else ".jpg"
        safe = "".join(c if c.isalnum() else "_" for c in current)[:40]
        fname = f"{counter:02d}_{safe}{ext}"
        with open(os.path.join(OUT_DIR, fname), "wb") as fh:
            fh.write(blob)
        manifest.setdefault(current, []).append(fname)

with open(MANIFEST, "w") as fh:
    json.dump(manifest, fh, indent=2)

print(f"Extracted {counter} screenshots to {OUT_DIR}")
print("--- section -> screenshots ---")
for sec, files in manifest.items():
    if files:
        print(f"\n[{sec}]")
        for f in files:
            print("   ", f)
