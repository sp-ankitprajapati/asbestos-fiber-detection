# Asbestos Fiber Mark Detection

Fine-tuning YOLOv8n to find four visual types of fiber marks in microscope
images, plus the checks needed to trust its numbers: a dataset audit that found
train/test leakage in the public benchmark, a leakage-reduced split, two kinds of
evaluation and a local review demo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sp-ankitprajapati/asbestos-fiber-detection/blob/main/notebooks/train_colab.ipynb)

> The labels **thick-dark**, **thick-light**, **thin-dark** and **thin-light**
> describe how a mark looks, not which asbestos mineral it is. The model can point
> a reviewer to marks worth checking; it cannot decide whether a material
> contains asbestos.

## Where this fits in an asbestos lab

Lab methods for building materials, such as JIS A 1481-1 (aligned with
ISO 22262-1), rely on a trained analyst at the microscope. The analyst
identifies asbestos from fiber shape and optical properties under polarized
light, such as refractive index, birefringence and extinction, and labs can
confirm results with XRD or SEM. Scanning many fields of view by eye is slow and
repetitive.

This project explores one small, assistive step: flagging fiber-like marks in an
image so a reviewer can look at the right places first. It does not identify
minerals or estimate asbestos content.

## Pipeline

```mermaid
flowchart LR
    A["RF100 Asbestos<br/>1,331 images"] --> B["Audit<br/>labels, duplicates,<br/>source names"]
    B --> C["Grouped split<br/>137 source groups"]
    C --> D["Fine-tune YOLOv8n<br/>from COCO weights"]
    D --> E["mAP and per-class AP<br/>(ranking quality)"]
    D --> F["Threshold sweep<br/>(precision / recall)"]
    F --> G["Gradio review demo"]
```

## Key finding: the official split leaks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/split_overlap_dark.png">
  <img src="assets/split_overlap.png" width="720" alt="129 source images appear in more than one official split: 71 in train, val and test; 46 in train and val only; 12 in train and test only.">
</picture>

Roboflow exports several variants of each source image, and they share a
**source name**: the file name before the `.rf.` suffix. In the official RF100
split, 129 source names appear in more than one split, and 83 of them reach both
train and test. A model trained there can see variants of its own test images,
so its test score can look better than its real performance. No two files are
byte-identical across splits, so a simple duplicate check misses this.

`make_grouped_split.py` puts every image that shares a source name, or has
identical bytes, into one group (a union-find over both keys), then assigns whole
groups to roughly 70/20/10 splits with seed 42. The 1,331 images collapse into
just **137 source groups**: 925 train, 273 validation and 133 test images, with
no source name in more than one split.

This is a partial safeguard. Images with different names can still come from the
same specimen, so it cannot prove specimen-level independence. It also replaces
the official benchmark split, so scores are not directly comparable with
published RF100 results. Full audit: [`reports/data_audit.json`](reports/data_audit.json).

## Data

[RF100 Asbestos](https://universe.roboflow.com/roboflow-100/asbestos): 1,331
images (932 train, 266 validation, 133 test) with 9,451 labeled marks. Credited
to Ahmad Rabiee and published by Roboflow 100 under CC BY 4.0; downloaded from
the [Libre-YOLO mirror](https://huggingface.co/datasets/Libre-YOLO/asbestos) at
pinned revision `24d72f5`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/class_balance_dark.png">
  <img src="assets/class_balance.png" width="720" alt="Labeled marks per class: thick-light 5,911; thick-dark 1,752; thin-dark 971; thin-light 817.">
</picture>

The classes are imbalanced: thick-light marks are 63% of all labels, and
thin-light marks have about 7x fewer examples. Expect weaker results on the thin
classes, which is why evaluation reports AP per class, not only an average.

## Results

**Pending full training.** No model has been fully trained yet, so there are no
accuracy numbers to report. The [training notebook](notebooks/train_colab.ipynb)
runs the whole experiment on a free Colab or Kaggle GPU and produces these:

| Trained and tested on | Test mAP50 | Test mAP50-95 |
| --- | --- | --- |
| Grouped split (main result) | pending | pending |
| Official RF100 split (comparison) | pending | pending |

Each number belongs with its evaluation file in `reports/results/`, which
records the checkpoint hash, dataset revision, split and settings.

**What has been checked so far.** A [one-epoch CPU run](reports/verification.json)
on a tiny subset (4 training images) at commit `f324361` exercised baseline and
frozen-backbone training, evaluation export, the threshold sweep and the demo end
to end. That run checks that the code works, not how accurate the model is. The demo's color handling has been
fixed since (see design decisions); unit tests cover that fix, but the end-to-end
demo check has not been re-run. All 12 unit tests pass.

## Run it

### On Colab or Kaggle (recommended)

Open the notebook with the badge above, choose a **T4 GPU**, and run the cells in
order. Checkpoints and results are saved to Google Drive, and a resume cell
covers disconnects. The same notebook runs on Kaggle: upload it, choose a GPU T4
accelerator, turn Internet on and use **Save & Run All**; results are saved in
the notebook's Output.

### Locally

Use Python 3.12. Install a [PyTorch build](https://pytorch.org/get-started/locally/)
for your CPU or GPU, then, from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py          # download pinned data and audit it
python make_grouped_split.py    # build the leakage-reduced split
```

Train (use `--device cpu` without a GPU; reduce `--batch` if memory is short):

```bash
python train.py --data asbestos/grouped/data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```

Training uses AdamW (learning rate 0.001), seed 42 and early stopping after 20
epochs without improvement, so 100 epochs is a maximum. The best checkpoint and
its metadata go to `weights/best.pt` and `weights/best.metadata.json`.
Variations:

```bash
# Freeze the backbone (first ten modules) to compare a lighter fine-tune
python train.py --data asbestos/grouped/data.yaml --freeze 10 --lr0 0.0005 --name frozen10 --output weights/frozen10.pt --device 0
# Continue an interrupted run
python train.py --resume runs_out/yolov8n_baseline/weights/last.pt --device 0
```

Evaluate on validation data, choose settings, and only then touch the test split:

```bash
python evaluate.py --data asbestos/grouped/data.yaml --split val --output reports/evaluation_val.json
python threshold_sweep.py --data asbestos/grouped/data.yaml
python evaluate.py --data asbestos/grouped/data.yaml --split test --output reports/evaluation_test.json
```

Demo, then open <http://127.0.0.1:7860>:

```bash
python app.py
```

Unit tests and figures:

```bash
python -m unittest discover -s tests -v
python make_figures.py
```

## Design decisions

- **Transfer learning.** Training starts from COCO-pretrained YOLOv8n (3.14M
  parameters), not random weights, because 925 training images are too few to
  learn general visual features from scratch. `--freeze 10` freezes the backbone
  and cuts trainable parameters to 1.87M. Whether that helps accuracy has to be
  measured on the same split and training budget.
- **Two kinds of evaluation.** `evaluate.py` reports mAP50, mAP50-95 and
  per-class AP, which score ranking quality across all confidence levels. Its
  summary precision and recall are Ultralytics' best-F1 values.
  `threshold_sweep.py` answers a reviewer's question instead: at one fixed
  confidence setting, how many marks are found (recall) and how many flags are
  wrong (precision)?
- **Matching written from scratch.** `detection_metrics.py` processes predictions
  in descending confidence; each one claims the unused same-class target with the
  highest IoU, and counts as correct only at IoU ≥ 0.5. Matching IoU (0.5) and
  NMS IoU (0.7) are different settings: NMS removes the model's duplicate boxes,
  while matching decides whether a box is right.
- **Validation chooses, test confirms.** Models and thresholds are chosen on
  validation data. The test split is evaluated once, after those choices are
  fixed.
- **Confidence is not probability.** A score of 0.8 is an uncalibrated model
  score, not an 80% chance that a mark is asbestos.
- **Color order in the demo.** Gradio provides RGB images, but Ultralytics reads
  NumPy input as BGR, OpenCV's order. The demo flips channels before prediction
  and flips the annotated output back for display; `tests/test_app.py` checks
  both directions.
- **Reproducibility.** The dataset revision is pinned, training is seeded, and
  every checkpoint gets a metadata file with its hash, settings and library
  versions.

## Limitations

- No accuracy results yet; see Results.
- Appearance labels only: no mineral identification, OCR, segmentation or
  asbestos-content estimate.
- Grouping by source name reduces leakage but cannot prove specimen-level
  independence.
- One public dataset. Results may not transfer to other microscopes,
  magnifications or sample preparations.
- Confidence scores are not calibrated probabilities.

## Files

| File | Purpose |
| --- | --- |
| `prepare_data.py`, `data_utils.py` | Pinned download, label validation and split audit |
| `make_grouped_split.py` | Leakage-reduced split by source name and exact copies |
| `train.py` | Fine-tuning, freezing, resume and checkpoint metadata |
| `evaluate.py` | mAP and per-class AP export |
| `threshold_sweep.py`, `detection_metrics.py` | Fixed-threshold precision, recall and F1 |
| `app.py` | Local Gradio review demo |
| `make_figures.py` | README charts from saved reports |
| `notebooks/train_colab.ipynb` | Full training and evaluation on a Colab or Kaggle GPU |
| `tests/` | Label validation, grouping, matching and demo checks |
| `reports/` | Data audit and execution check |

Model weights, downloaded images and training runs stay out of Git.

## Attribution and dependencies

Dataset: [Roboflow 100 / Asbestos](https://universe.roboflow.com/roboflow-100/asbestos),
original contributor Ahmad Rabiee, CC BY 4.0, accessed through the Libre-YOLO
mirror; see the [RF100 benchmark paper](https://arxiv.org/abs/2211.13523).
Model and training library: [Ultralytics](https://github.com/ultralytics/ultralytics),
available under AGPL-3.0 or an enterprise license. Dataset and dependency licenses
apply independently. No dataset images or pretrained weights are stored here; if
you commit the notebook's sample-prediction figure, it shows dataset images under
the CC BY 4.0 license and credit above.
