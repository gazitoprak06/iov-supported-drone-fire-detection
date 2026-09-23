# UAV Fire Detection: Null-Baseline Auditing, the Location of Failure in the Decision Rule, and a Deep Edge Replacement

Reproduction package for the manuscript of the same name (Alaeddin Gazi Toprak,
Alireza Souri). The audited manuscript source is `makale.html`.

The paper implements a CPU-only, hand-set colour-space fire detector (V1),
audits it against constant-predictor null baselines, locates its failure in the
decision rule rather than in the evidence by fitting learned models on the same
features (V2), and then replaces the detector with a retrained MobileNetV3-Small
evaluated on a clip-level partition of a 483-clip real aerial corpus (the deep
edge detector).

The negative results are not hidden. On the still-image split the hand-set rule
is beaten on F1 by a constant fire predictor and on accuracy by a constant
no-fire predictor, across all seven colour-space ablations and all nine
threshold settings. That is the paper's primary result.

## What is in this repository

| Path | Contents |
| --- | --- |
| `v1_heuristic.py` | The per-frame pipeline of Sections 3.1 to 3.3 (Algorithm 1). Every experiment that exercises the hand-set detector calls this one function |
| `v2_learned_baselines.py` | Section 4.4 and Table 4: fits on train, selects on val, evaluates test once, in one program |
| `train_v3.py`, `extract_training_frames.py` | The deep edge detector: frame extraction by clip partition, then fine-tuning |
| `evaluate_v3_videos.py` | PyTorch reference clip evaluator |
| `tools/` | Shared metric modules, the torch-free evaluator, the benchmarks and the audit gates (see below) |
| `Proje_Kodlari/evaluation_results/` | Every reported number, as the artifact that produced it |
| `Proje_Kodlari/annotations/`, `labels.json` | Clip manifest, per-clip labels, SHA-256 hashes, cohort assignments |
| `MANIFEST.sha256` | Integrity list for the release set |

### Shared definition modules

Two modules exist so that programs cannot drift apart in how they count:

- `tools/clip_metrics.py` — the alarm rule, the sampling policy, the clip
  grouping and the Wilson interval. Both clip evaluators and the viewer import
  from it rather than restating it. The four constants of Algorithm 2 in the
  manuscript — the fire class index, two samples per second, three consecutive
  samples for an alarm and the 24 fps fallback — are defined here and nowhere
  else.
- `tools/image_metrics.py` — the split walker, the confusion-matrix arithmetic,
  the null baselines and the McNemar test for the still-image cohort.

### Which program produced which table

| Table / Section | Program | Artifact |
| --- | --- | --- |
| Section 4.1, Table 1 | `tools/eval_image_level.py` | `v1_image_level/v1_image_level_metrics.json` |
| Table 2 | `tools/resolution_control.py` | `v1_image_level/v1_resolution_control.json` |
| Table 3 | `tools/saturation_sweep.py` | `v1_image_level/v1_saturation_sweep.json` |
| Table 4 | `v2_learned_baselines.py` | `v2_baselines/v2_baseline_metrics.json` |
| Section 4.5, Table 5 | `tools/eval_clips_numpy.py` (Algorithm 2) | `v3_deep_edge/v3_results_by_split.json` |
| Fig. 6 | `tools/make_confusion_figure.py` | `v3_confusion_matrix.png` |
| Latency, deep edge | `tools/benchmark_latency.py` | `v3_deep_edge/v3_latency.json` |
| Latency variability, V1 (Section 5) | `tools/benchmark_v1_latency.py` | `v1_image_level/v1_latency_variability.json` |
| Source and event controls (Section 4.5) | `tools/corpus_controls.py` | `v3_deep_edge/v3_corpus_controls.json` |

`tools/box_count_stages.py` is a diagnostic rather than a result: it reports the
spurious-box total at each stage of the localization chain, so that the totals
quoted in Section 4.3 can be attributed to a stage rather than guessed.

## Install

Two environments are recorded in the manuscript, and the header of every result
file names the one that produced it. Environment A is the Windows 10 / Python
3.12.7 target machine; Environment B is a containerised Linux host with Python
3.10.12 and OpenCV 4.13.0. The released image-level artifacts are the
Environment B recomputation, which reproduces the Environment A figures on all
410 test images and all seven ablation rows.

The audited target environment is Windows 10 with Python 3.12.7. Create a
fresh virtual environment for it rather than reusing one that happens to sit in
the project tree:

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The `--index-url` matters: the default index serves a CUDA build, which is
several gigabytes and unnecessary here, and the figures in this paper are
CPU-only by construction.

PyTorch is needed only for training and for the reference clip evaluator. The
clip-level verdicts reported in the paper were produced by the torch-free NumPy
path, so Section 4.5 can be reproduced without installing torch.

## Verify the recorded results

These gates need no external data and should be run first:

```
python -m unittest discover -s tests
python tools/check_paper_numbers.py
python tools/check_no_data_lost.py <before.html> makale.html
```

The test suite covers the two shared definition modules against values worked
out by hand, and asserts that the image-level and clip-level modules agree on
every cell they both compute.

`check_paper_numbers.py` parses the manuscript, matches every tagged number
against the artifact that produced it to the displayed precision, recomputes the
Wilson intervals independently, verifies that the three clip partitions sum to
the corpus on every confusion-matrix cell, reconciles the results artifact
against the clip manifest group by group, and fails if any superseded figure
survives anywhere in the file. It exits non-zero, so it can gate a commit.

## Reproduce from source data

### Still images (Sections 4.1 to 4.4)

The image dataset is not redistributed here. Download *The Wildfire Dataset*
(CC BY 4.0) from Kaggle and place it as:

- **The Wildfire Dataset.** El-Madafri I, Peña M, Olmedo-Torre N. *Forests.*
  2023;14(9):1697. doi:10.3390/f14091697 — Licence **CC BY 4.0**.
  <https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset>


```
Proje_Kodlari/data/AR Souri Dataset/{train,val,test}/{fire,nofire}/
```

The study used 2699 supported JPG/PNG files: 1887 train, 402 validation, 410
test. Then:

```
python tools/verify_v1_equivalence.py     # proves the parameterized pipeline
                                          # reduces to the released function
python tools/eval_image_level.py          # Section 4.1 and Table 1
python tools/resolution_control.py        # Table 2
python tools/saturation_sweep.py          # Table 3
python v2_learned_baselines.py            # Table 4
python tools/benchmark_v1_latency.py      # Section 5 variability record
```

The first three run the morphological chain at native resolution over images
spanning three orders of magnitude, so a full pass is long. Each checkpoints to
disk and resumes; pass `--max-seconds S` to bound a single invocation and rerun
until it reports completion.

One convention of Table 1 is worth knowing before rerunning it. The white-hot
bypass of Section 3.1 is itself an RGB-channel rule, so it is disabled together
with the RGB mask (`--whitehot follows_rgb`, the default). `--whitehot always`
keeps it on in every row instead and changes the three rows in which RGB is off;
both are released, the second under a `_whitehot_always` suffix.

### Clips (Section 4.5)

The 483-clip corpus was not collected by the authors and is not redistributed.
It is assembled from two public sources, both cited in the manuscript:

- **ERA — Event Recognition in Aerial videos.** Mou L, Hua Y, Jin P, Zhu XX.
  *IEEE Geosci Remote Sens Mag.* 2020;8(4):125-33. doi:10.1109/mgrs.2020.3005751
  Licence **CC BY-NC 4.0** (non-commercial).
  <https://lcmou.github.io/ERA_Dataset/>
  Supplies all 313 negative clips and 84 of the 170 positives. Clip filenames
  carry the ERA event category as a prefix (`Fire_`, `Boating_`, `Soccer_`, ...).

- **Boreal Forest Fire — UAV-collected wildfire detection and smoke
  segmentation.** Pesonen J, Raita-Hakola AM, Joutsalainen J, Hakala T,
  Akhtar W, Koivumäki N, et al. *Sci Data.* 2025;12:1428.
  doi:10.1038/s41597-025-05634-0 — Licence **CC BY 4.0**.
  <https://etsin.fairdata.fi/dataset/1dce1023-493a-4d63-a906-f2a44f831898>
  Supplies the remaining 86 positives, filmed by drone over prescribed burns at
  Evo, Heinola, Karkkila and Ruokolahti in Finland; those four municipality
  names are the filename prefixes the source control of Section 4.5 keys on. `Proje_Kodlari/annotations/video_evaluation_manifest.json`
gives the label and the train/monitor/held-back assignment of every clip, and
`labels.json` gives per-clip SHA-256 hashes so that a local copy can be checked
against the one used here. Frames extracted from the corpus are likewise not
redistributed; rebuild them with `extract_training_frames.py`.

```
python extract_training_frames.py
python train_v3.py
python tools/eval_clips_numpy.py          # the path that produced the figures
python evaluate_v3_videos.py              # PyTorch reference path
python tools/verify_numpy_model.py        # cross-checks the two, frame by frame
python tools/corpus_controls.py           # source and event-level controls
python tools/multiseed_v3.py --seeds 0 1 2 3 4   # seed sweep (about an hour per seed)
```

Rerunning any of these overwrites the recorded artifacts.

## Scope and limitations

This is not a real flight trial and there is no physical V2X or radio field
validation. The image cohorts carry no bounding-box annotations, so every
image-level result is classification and no IoU or mAP is computed anywhere.
The learned models of Section 4.4 are classifiers and do not localize. The deep
edge run is three epochs on a single seed, and its 194 unseen clips are few
enough that the 95% Wilson interval on the recall spans [83.91%, 96.82%].
Section 5 states the full set.

## Licence and citation

Source code is released under the MIT licence (`LICENSE`). That licence does not
relicense the image dataset or the clip corpus, which carry their own terms.
Software citation metadata is in `CITATION.cff`.
