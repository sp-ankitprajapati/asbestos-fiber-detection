# Asbestos Fiber Mark Detection

YOLOv8n transfer learning for detecting four visual categories of fiber marks in
microscope images. Includes a local Gradio demo, dataset audit, evaluation export
and a confidence-threshold sweep.

The detector labels **thick-dark**, **thick-light**, **thin-dark** and **thin-light**
marks. These are appearance labels, not asbestos mineral types. Predictions can
assist image review; they do not establish whether a material contains asbestos.

## Data

The [RF100 Asbestos dataset](https://universe.roboflow.com/roboflow-100/asbestos)
contains 1,331 images: 932 training, 266 validation and 133 test images.
The original dataset is credited to Ahmad Rabiee; the RF100 version is published
by Roboflow 100 under CC BY 4.0. Images and annotations are downloaded separately
from the [Libre-YOLO mirror](https://huggingface.co/datasets/Libre-YOLO/asbestos),
at revision `24d72f5484922bf95281cd9a6a9ef84760cd3d0c`.

The audit checks label ranges, class counts, identical file hashes across splits
and repeated source names. Exact-hash checks cannot rule out resized or augmented
copies; source-name overlaps need review before interpreting held-out scores.

**Split audit:** 129 source names occur in multiple supplied splits. Use the
grouped split below for experiments instead of assuming the supplied test split
is independent. It joins matching source names and identical file hashes before
assigning approximately 70/20/10 splits with seed 42. Different names can still
refer to related specimens or synthetic variants, so this is a partial safeguard,
not proof of specimen-level independence. It changes the RF100 benchmark split.
The audit found no identical image-file hashes across the supplied splits.
The regrouped dataset contains 925 train / 273 validation / 133 test images in
137 source groups. See the [full audit](reports/data_audit.json).

## Setup

Use Python 3.12 and a virtual environment. Install a suitable
[PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or CUDA GPU,
then install the project dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
python make_grouped_split.py
```

Commands below run from the repository root. Data downloads require internet
access. Training initially downloads the COCO-pretrained `yolov8n.pt` checkpoint.

## Train

```bash
python train.py --data asbestos/grouped/data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```

For CPU use `--device cpu`. Reduce the batch size if GPU memory is limited.
Training uses AdamW with initial learning rate 0.001, seed 42 and early stopping
after 20 epochs without improvement. **100 is a maximum, not a promise that every
run completes 100 epochs.** The selected checkpoint and its settings/hash are
copied to `weights/best.pt` and `weights/best.metadata.json`.

To compare a smaller fine-tuning change, freeze the first ten model modules:

```bash
python train.py --data asbestos/grouped/data.yaml --freeze 10 --lr0 0.0005 --name frozen10 --output weights/frozen10.pt --device 0
```

Use the same split, image size, seed and training budget for comparisons. Freezing
reduces trainable parameters; it may help or hurt accuracy and must be measured.
The first ten modules are the backbone in the selected YOLOv8n architecture.

Resume an **interrupted** run with its `last.pt`:

```bash
python train.py --resume runs_out/yolov8n_baseline/weights/last.pt --device 0
```

Resume restores the checkpoint's saved training settings and optimizer state.
On Colab, mount Drive and use `--project /content/drive/MyDrive/asbestos_runs` to
keep epoch checkpoints when the runtime disconnects.

## Evaluate

```bash
python evaluate.py --data asbestos/grouped/data.yaml --split val --output reports/evaluation_val.json
python threshold_sweep.py --data asbestos/grouped/data.yaml
# After choosing the model/settings on validation data:
python evaluate.py --data asbestos/grouped/data.yaml --split test --output reports/evaluation_test.json
```

`evaluate.py` exports mAP50, mAP50–95 and per-class AP using a low confidence
cutoff (0.001). Its summary precision/recall are Ultralytics' best-F1 values.
The [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/)
explains this distinction.

`threshold_sweep.py` instead reports **micro precision, recall, F1, TP, FP and FN
at each actual confidence threshold** on validation data. A prediction matches
one unused same-class target at IoU ≥ 0.5, processed in descending confidence.
NMS IoU stays at 0.7. Matching IoU and NMS IoU serve different purposes.
Ratios with zero denominators are recorded as zero. This sweep does not calculate
AP; filtering away low-score predictions can truncate the precision-recall curve.

Keep test data out of model and threshold selection. Choose an operating point
on validation data for the review workload, then evaluate the fixed choice on
an independent test set. A higher threshold does not guarantee higher precision.

## Demo

```bash
python app.py
```

Open <http://127.0.0.1:7860>, upload a microscope image and adjust the confidence
threshold. The demo shows boxes, class labels, scores and a detection table.
It requires trained weights and gives a clear error if they are missing.
Confidence is a model score, not a calibrated probability of asbestos presence.

## Results and limitations

No trained checkpoint or reproducible full-training evaluation is distributed
with this repository. Report model performance only alongside the checkpoint
hash, dataset revision, split, settings and exported evaluation file.

A short execution check verifies the code path; it is not an accuracy benchmark.
The [verification report](reports/verification.json) records a one-epoch CPU
execution check for baseline and frozen-backbone training, evaluation export,
threshold counts and the local demo, plus 11 unit tests. Full training and
independent accuracy evaluation remain to be completed.

This project demonstrates detection and CNN transfer learning. It does not
implement OCR, drawing-layout analysis, segmentation or mineral identification.
Published results from other datasets or lab systems are not directly comparable.
Before claiming generalization, inspect duplicates, source/specimen grouping,
class imbalance and performance on genuinely independent images.

## Files

| File | Purpose |
| --- | --- |
| `prepare_data.py`, `data_utils.py` | Pinned download, annotation and split checks |
| `make_grouped_split.py` | Reproducible source-name and exact-copy grouping |
| `train.py` | Fine-tuning, freezing, resume and checkpoint metadata |
| `evaluate.py` | Standard AP and per-class evaluation |
| `threshold_sweep.py`, `detection_metrics.py` | Fixed-threshold operating counts |
| `app.py` | Local image review demo |
| `tests/` | Matching and dataset validation checks |

Run the lightweight checks with `python -m unittest discover -s tests -v`.
Model weights, downloaded images and generated training runs stay out of Git.

## Attribution and dependencies

Dataset: [Roboflow 100 / Asbestos](https://universe.roboflow.com/roboflow-100/asbestos),
original contributor Ahmad Rabiee, CC BY 4.0; accessed through the Libre-YOLO mirror.
[RF100 benchmark paper](https://arxiv.org/abs/2211.13523).
Model/training library: [Ultralytics](https://github.com/ultralytics/ultralytics),
which provides AGPL-3.0 and enterprise licensing options. Dataset and dependency
licenses continue to apply independently; no dataset images or pretrained weights
are redistributed here.
