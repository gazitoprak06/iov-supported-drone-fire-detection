"""Guard for condensing passes: no measured value may vanish from the paper.

Condensing prose is safe; dropping a number is not. Run this with a snapshot of
the manuscript taken before the edit:

    python tools/check_no_data_lost.py makale_before_trim.html makale.html

It extracts every numeric token from both files and reports any that the old
version contained and the new one does not. Section, table, figure and
reference numbers are ignored, since renumbering is legitimate; everything that
looks like a measurement is not.
"""

import html
import re
import sys
from pathlib import Path

# Tokens that are structure rather than data.
IGNORE_CONTEXT = re.compile(
    r"(Section|Sections|Table|Tables|Figure|Figures|Fig\.|Figs\.|Eq\.|Eqs\.|"
    r"reference|references)\s*$", re.I)


def numbers(path):
    raw = Path(path).read_text(encoding="utf-8")
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"\s+", " ", text)
    found = {}
    for m in re.finditer(r"\d+\.\d+%?|\d+%|\b\d{2,}\b", text):
        tok = m.group(0)
        before = text[max(0, m.start() - 12):m.start()]
        if IGNORE_CONTEXT.search(before):
            continue
        if tok.rstrip("%").replace(".", "").isdigit() and "." not in tok and not tok.endswith("%"):
            if len(tok) <= 2:            # small bare integers are prose, not data
                continue
        found.setdefault(tok, 0)
        found[tok] += 1
    return found


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    old, new = numbers(sys.argv[1]), numbers(sys.argv[2])

    vanished = sorted(t for t in old if t not in new)
    reduced = sorted((t, old[t], new[t]) for t in old if t in new and new[t] < old[t])

    print(f"old: {len(old)} distinct numeric tokens; new: {len(new)}")
    if vanished:
        print(f"\nGONE ENTIRELY ({len(vanished)}) — each must be deliberate:")
        for t in vanished:
            print(f"  {t}  (appeared {old[t]}x)")
    else:
        print("\nNo measured value disappeared from the manuscript.")

    if reduced:
        print(f"\nFewer occurrences (fine if it was repetition) — {len(reduced)}:")
        for t, a, b in reduced[:40]:
            print(f"  {t}: {a} -> {b}")

    sys.exit(1 if vanished else 0)


if __name__ == "__main__":
    main()
