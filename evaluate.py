"""Export standard AP metrics and per-class results for a trained checkpoint."""
import argparse
from importlib.metadata import version
from pathlib import Path

from data_utils import CLASS_NAMES, load_dataset, sha256, validate_model_names, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--data", default="asbestos/data.local.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.error(f"Missing {args.weights}. Train first or supply your trained checkpoint.")
    load_dataset(args.data)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    validate_model_names(model)
    metrics = model.val(data=str(Path(args.data).resolve()), split=args.split,
                        imgsz=args.imgsz, batch=args.batch, device=args.device,
                        conf=0.001, iou=0.7, max_det=300, rect=False, workers=0,
                        plots=True, project=str(Path("runs_out/evaluation").resolve()),
                        name=args.split)
    per_class = []
    for index, cls in enumerate(metrics.box.ap_class_index):
        per_class.append({"class": CLASS_NAMES[int(cls)],
                          "map50": float(metrics.box.ap50[index]),
                          "map50_95": float(metrics.box.ap[index])})
    write_json(args.output, {"split": args.split, "weights_sha256": sha256(args.weights),
        "data_yaml_sha256": sha256(args.data), "settings": vars(args),
        "ultralytics_version": version("ultralytics"),
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "precision_at_best_f1": float(metrics.box.mp), "recall_at_best_f1": float(metrics.box.mr),
        "per_class": per_class,
        "note": "Summary precision/recall use Ultralytics best-F1 operating point. "
                "Use threshold_sweep.py for fixed-threshold micro precision/recall."})
    print("Saved", args.output)


if __name__ == "__main__":
    main()
