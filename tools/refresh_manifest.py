"""Bring MANIFEST.sha256 back into agreement with the release.

The manifest is the integrity list a reader checks a local copy against, so a
stale entry is worse than no entry: it reports corruption where there is none,
or misses it where there is. Every file git tracks belongs in it, and nothing
else does.

The entries keep the Windows-style relative paths the manifest was first written
with, so that a refresh produces a diff of the digests that changed rather than
a rewrite of every line.

Usage:  python tools/refresh_manifest.py            # report only
        python tools/refresh_manifest.py --write    # rewrite the manifest
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"


def entry_path(field: str) -> Path:
    p = Path(field.strip().lstrip("*").replace("\\", "/"))
    if p.parts and p.parts[0] == ".":
        p = Path(*p.parts[1:])
    return ROOT / p


def to_field(rel: Path) -> str:
    return ".\\" + str(rel).replace("/", "\\")


def tracked() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv) -> int:
    write = "--write" in argv
    existing = {}
    order = []
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            h, f = line.split(None, 1)
            existing[str(entry_path(f).relative_to(ROOT)).replace("\\", "/")] = (h, f.strip())
            order.append(str(entry_path(f).relative_to(ROOT)).replace("\\", "/"))

    files = tracked()
    keys = {str(p).replace("\\", "/") for p in files}
    # the manifest lists itself in no sensible way
    keys.discard("MANIFEST.sha256")

    added = sorted(keys - set(existing))
    removed = sorted(set(existing) - keys)
    changed, ok, missing = [], 0, []

    lines = []
    for k in [k for k in order if k in keys] + added:
        p = ROOT / k
        if not p.exists():
            missing.append(k)
            continue
        d = digest(p)
        if k in existing and existing[k][0] != d:
            changed.append(k)
        elif k in existing:
            ok += 1
        field = existing[k][1] if k in existing else to_field(Path(k))
        lines.append(f"{d}  {field}")

    print(f"tracked files : {len(keys)}")
    print(f"unchanged     : {ok}")
    print(f"digest changed: {len(changed)}" + (f"  {changed}" if changed else ""))
    print(f"newly added   : {len(added)}" + (f"  {added}" if added else ""))
    print(f"no longer tracked: {len(removed)}" + (f"  {removed}" if removed else ""))
    if missing:
        print(f"MISSING ON DISK  : {missing}")
        return 1

    if write:
        MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {MANIFEST.name} with {len(lines)} entries")
    else:
        print("\nreport only; pass --write to rewrite the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
