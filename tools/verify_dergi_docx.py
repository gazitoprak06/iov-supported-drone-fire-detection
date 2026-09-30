"""Final sanity check on makale_dergi.docx after fix_dergi_docx.py (+ the two
follow-ups): confirms word/document.xml is still well-formed XML, and scans
for leftover instances of every bug the advisor's report and our own read-
through found.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "makale_dergi.docx"
DOC_PART = "word/document.xml"
W_T = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
_NUM_ENTITY = re.compile(r"&#x([0-9a-fA-F]+);|&#(\d+);")


def unescape(s):
    s = _NUM_ENTITY.sub(lambda m: chr(int(m.group(1), 16) if m.group(1) else int(m.group(2))), s)
    return (s.replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&"))


def main():
    with zipfile.ZipFile(SRC, "r") as zin:
        bad = zin.testzip()
        if bad:
            print(f"CORRUPT ZIP ENTRY: {bad}")
            return 1
        xml_bytes = zin.read(DOC_PART)

    print("1. zip integrity: ok (testzip found no bad entries)")

    try:
        ET.fromstring(xml_bytes)
        print("2. word/document.xml is well-formed XML: ok")
    except ET.ParseError as exc:
        print(f"2. word/document.xml is NOT well-formed XML: {exc}")
        return 1

    flat = "".join(unescape(m.group(1)) for m in W_T.finditer(xml_bytes.decode("utf-8")))

    checks = [
        ("Algorithm 1 table present", "Algorithm 1: Per-frame detection pipeline" in flat, True),
        ("Eq. (2) present",
         ("altitude-scaled minimum cluster area" in flat) or ("A_min(h) = 600" in flat), True),
        ("citation [30] in body", flat.count("[30]") >= 1, True),
        ("citation [31] in body", flat.count("[31]") >= 1, True),
        ("citation [33] in body", flat.count("[33]") >= 1, True),
        ("no orphan 'Section 3.4'", "Section 3.4" in flat, False),
        ("no leftover '(§4.4)'", "(§4.4)" in flat, False),
        ("no leftover '(§4.5)'", "(§4.5)" in flat, False),
        ("no bare 'OpenCV 5.0 '", re.search(r"OpenCV 5\.0[^.]", flat) is not None, False),
        ("no bare 'OpenCV 4.13 '", re.search(r"OpenCV 4\.13[^.]", flat) is not None, False),
        ("no 'labelled'", "labelled" in flat, False),
        ("no 'neutralised'", "neutralised" in flat, False),
        ("Table 1 HSV value corrected", "0 ≤ H ≤ 60" in flat, False),
        ("Table 1 RGB value corrected", "R > 120" in flat, False),
        ("abstract run-on split", "degenerate one, whereas a model" in flat, False),
        ("contribution-4 run-on split",
         "First, the rule cannot be repaired within its own family: a saturation" in flat, False),
        ("Figure 6 disambiguation note present", "not read from this panel" in flat, True),
    ]

    print()
    all_ok = True
    for label, found, want_found in checks:
        ok = (found == want_found)
        all_ok &= ok
        state = "ok" if ok else "PROBLEM"
        print(f"  {state:8} {label}")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see PROBLEM lines above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
