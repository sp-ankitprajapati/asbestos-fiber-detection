"""Measure operating-point precision/recall on validation images (not mAP)."""
import argparse
import csv
from pathlib import Path

from data_utils import read_labels, sha256, split_images, validate_model_names, write_json
from detection_metrics import match_counts, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--data", default="asbestos/data.local.yaml")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75])
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="reports/threshold_sweep.csv")
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.error(f"Missing {args.weights}. Train first or supply your trained checkpoint.")
    if not all(0 < c < 1 for c in args.thresholds) or not 0 < args.match_iou <= 1 or not 0 < args.nms_iou <= 1:
        parser.error("thresholds must be in (0,1); IoU values must be in (0,1]")

    from ultralytics import YOLO

    images = split_images(args.data, "val")
    model = YOLO(args.weights)
    validate_model_names(model)
    thresholds = sorted(set(args.thresholds))
    counts = {conf: [0, 0, 0] for conf in thresholds}
    # Infer once at the lowest score; apply each operating threshold to that output.
    results = model.predict(source=[str(p) for p in images], stream=True, conf=min(thresholds),
                            iou=args.nms_iou, imgsz=args.imgsz, max_det=args.max_det,
                            device=args.device, verbose=False, rect=False)
    seen = 0
    for result in results:
        targets = read_labels(Path(result.path))
        boxes = result.boxes
        predictions = list(zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()))
        for conf in thresholds:
            values = match_counts(predictions, targets, conf, args.match_iou)
            counts[conf] = [a + b for a, b in zip(counts[conf], values)]
        seen += 1
    if seen != len(images):
        raise RuntimeError(f"Expected {len(images)} predictions, received {seen}")
    rows = [{"confidence": conf, **summarize(*counts[conf])} for conf in thresholds]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output.with_suffix(".metadata.json"), {
        "split": "val", "images": seen, "weights_sha256": sha256(args.weights),
        "data_yaml_sha256": sha256(args.data), "settings": vars(args),
        "matching": "class-aware, confidence-ordered, greedy IoU; micro counts",
        "undefined_ratios": "reported as zero", "rows": rows})
    for row in rows:
        print(row)
    print("Saved", output)


if __name__ == "__main__":
    main()
