import collections
import os
import json
import cv2
from pathlib import Path

def main():
    root_dir = Path("Proje_Kodlari")
    manifest_path = root_dir / "annotations" / "video_evaluation_manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    videos = manifest['videos']
    dataset_out = root_dir / "data" / "dataset"
    FRAMES_PER_VIDEO = 10

    # Manifest split -> image folder. The manifest carries three split values;
    # 'validation' is deliberately held back and contributes no frames to any
    # folder, so those clips stay untouched evidence for the final evaluation.
    #
    # This mapping used to be an inline `if split == 'test': split = 'val'`,
    # which routed 'validation' to a directory that was never created. cv2
    # returns False rather than raising when the target directory is absent, so
    # those clips were dropped without a single line of output. The explicit
    # table plus the counters below exist so that can never recur silently.
    SPLIT_TO_FOLDER = {
        'train': 'train',       # weight updates
        'test': 'val',          # per-epoch monitor, no gradients
        'validation': None,     # held back entirely
    }

    print(f"Extracting max {FRAMES_PER_VIDEO} frames per video into {dataset_out}", flush=True)

    for folder in ('train', 'val'):
        for label in ('fire', 'no_fire'):
            os.makedirs(dataset_out / folder / label, exist_ok=True)

    frames_extracted = 0
    per_split = collections.Counter()
    missing_files = []
    undecodable = []
    short_clips = []
    unknown_splits = collections.Counter()

    for i, v in enumerate(videos):
        v_path = root_dir / v['path']
        if not v_path.exists():
            missing_files.append(v['path'])
            continue

        split = v.get('split', 'train')
        if split not in SPLIT_TO_FOLDER:
            unknown_splits[split] += 1
            continue

        folder = SPLIT_TO_FOLDER[split]
        if folder is None:
            per_split[f'{split} (held back)'] += 1
            continue

        label = v['label']
        out_folder = dataset_out / folder / label
        per_split[f'{split} -> {folder}'] += 1

        cap = cv2.VideoCapture(str(v_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            per_split[f'{split} -> {folder}'] -= 1
            undecodable.append(v['path'])
            continue

        step = max(1, total_frames // FRAMES_PER_VIDEO)
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= FRAMES_PER_VIDEO:
                break
            
            if frame_idx % step == 0:
                frame = cv2.resize(frame, (224, 224))
                img_name = f"{v['id']}_frame_{frame_idx:04d}.jpg"
                # cv2.imwrite returns False instead of raising when the target
                # directory does not exist. That silent failure is what dropped
                # an entire split before; never ignore this return value.
                if not cv2.imwrite(str(out_folder / img_name), frame):
                    raise SystemExit(f"ERROR: cv2.imwrite failed for {out_folder / img_name}; "
                                     f"does {out_folder} exist and is it writable?")
                frames_extracted += 1
                saved_count += 1

            frame_idx += 1

        cap.release()
        if saved_count < FRAMES_PER_VIDEO:
            short_clips.append((v['path'], saved_count))
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(videos)} videos. Extracted {frames_extracted} frames.", flush=True)

    print(f"\nDone! Total extracted frames: {frames_extracted}", flush=True)
    print("Clips routed per split:")
    for key, count in sorted(per_split.items()):
        print(f"  {key:<28} {count:>4}")
    # A clip that is missing or undecodable silently shrinks the training set,
    # which would change every reported number without changing any visible line
    # of this script. It is therefore a hard failure, not a warning.
    if missing_files or undecodable:
        if missing_files:
            print(f"\nERROR: {len(missing_files)} manifest clips are missing on disk, "
                  f"e.g. {missing_files[:3]}")
        if undecodable:
            print(f"\nERROR: {len(undecodable)} clips reported no frames, "
                  f"e.g. {undecodable[:3]}")
        raise SystemExit(
            "Refusing to proceed: the extracted set would not match the manifest. "
            "Restore the missing or unreadable clips, or amend the manifest, and rerun."
        )
    if short_clips:
        print(f"\nNOTE: {len(short_clips)} clips yielded fewer than {FRAMES_PER_VIDEO} frames, "
              f"e.g. {short_clips[:3]}")
    if unknown_splits:
        raise SystemExit(
            f"\nERROR: manifest contains split value(s) {dict(unknown_splits)} that are not "
            f"in SPLIT_TO_FOLDER. Add them explicitly rather than letting them be dropped; "
            f"tools/clip_metrics.py:SPLIT_ROLE must be updated to match."
        )

if __name__ == "__main__":
    main()
