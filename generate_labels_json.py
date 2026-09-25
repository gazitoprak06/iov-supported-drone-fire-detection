"""Generate labels.json from the video manifest and on-disk video files.

labels.json is cited in the Data Availability Statement:
    "Per-clip labels, SHA-256 hashes and cohort assignments are released as labels.json."

The file contains, for every clip in the 483-clip corpus:
  - id           : clip identifier from the manifest
  - label        : 'fire' or 'no_fire'
  - split        : 'train', 'test', or 'validation' (manifest field)
  - role         : derived role (gradient_train / epoch_monitor / untouched)
  - sha256       : SHA-256 hex digest of the raw video file
  - path         : relative path from the Proje_Kodlari directory

Usage:
    python generate_labels_json.py

Output: labels.json in the project root.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "Proje_Kodlari"
MANIFEST = ROOT / "annotations" / "video_evaluation_manifest.json"
OUT = Path("labels.json")

SPLIT_ROLE = {
    "train":      "gradient_train",
    "test":       "epoch_monitor",
    "validation": "untouched",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = manifest["videos"]
    print(f"Processing {len(videos)} clips...", flush=True)

    records = []
    missing = []

    for i, v in enumerate(videos):
        vid_path = ROOT / v["path"]
        if not vid_path.exists():
            missing.append(v["path"])
            records.append({
                "id":     v["id"],
                "label":  v["label"],
                "split":  v.get("split", "train"),
                "role":   SPLIT_ROLE.get(v.get("split", "train"), "unknown"),
                "sha256": None,
                "path":   v["path"],
                "note":   "file not found on disk at generation time",
            })
            continue

        digest = sha256_file(vid_path)
        records.append({
            "id":     v["id"],
            "label":  v["label"],
            "split":  v.get("split", "train"),
            "role":   SPLIT_ROLE.get(v.get("split", "train"), "unknown"),
            "sha256": digest,
            "path":   v["path"],
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(videos)}]", flush=True)

    if missing:
        print(f"\nWARNING: {len(missing)} clip files were not found on disk.")
        for p in missing[:5]:
            print(f"  {p}")

    # Summary statistics
    by_label = {}
    by_split = {}
    for r in records:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1

    payload = {
        "schema_version": "1.0",
        "description": (
            "Per-clip labels, SHA-256 hashes and cohort assignments for the "
            "483-clip aerial corpus used in Section 4.8."
        ),
        "summary": {
            "total_clips": len(records),
            "by_label": by_label,
            "by_split": by_split,
        },
        "clips": records,
    }

    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}  ({len(records)} clips)")
    print(f"  by label: {by_label}")
    print(f"  by split: {by_split}")


if __name__ == "__main__":
    main()
