"""Fine-tune a COCO-pretrained YOLOv8n detector on fiber-mark labels."""
import argparse
from importlib.metadata import version
from pathlib import Path
import shutil

from data_utils import DATASET_ID, DATASET_REVISION, load_dataset, sha256, write_json
from prepare_data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Existing data YAML; otherwise download the pinned dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model modules")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu or CUDA device, e.g. 0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs_out")
    parser.add_argument("--name", default="yolov8n_baseline")
    parser.add_argument("--output", default="weights/best.pt")
    parser.add_argument("--resume", help="Resume an interrupted run from its last.pt")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.freeze < 0 or args.lr0 <= 0:
        parser.error("epochs, imgsz, batch and lr0 must be positive; freeze must be nonnegative")

    from ultralytics import YOLO

    parameter_counts = {}

    def record_parameters(trainer):
        parameter_counts.update(
            total=sum(p.numel() for p in trainer.model.parameters()),
            trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        )

    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.is_file():
            parser.error(f"Resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.add_callback("on_train_start", record_parameters)
        model.train(resume=True, device=args.device, workers=args.workers)
        data = Path(model.trainer.args.data)
    else:
        data = Path(args.data) if args.data else prepare_data()
        load_dataset(data)
        model = YOLO("yolov8n.pt")
        model.add_callback("on_train_start", record_parameters)
        model.train(data=str(data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, patience=args.patience, optimizer="AdamW", lr0=args.lr0,
                    freeze=args.freeze, seed=args.seed, deterministic=True,
                    device=args.device, workers=args.workers,
                    project=str(Path(args.project).resolve()), name=args.name)

    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt: {best}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() != output.resolve():
        shutil.copy2(best, output)
    actual = vars(model.trainer.args)
    manifest = {"weights_sha256": sha256(output), "data_yaml_sha256": sha256(data),
                "run_directory": str(model.trainer.save_dir),
                "training_args": actual,
                "parameters_at_training_start": parameter_counts,
                "versions": {p: version(p) for p in ("ultralytics", "torch", "torchvision")}}
    if not args.data and not args.resume:
        manifest["dataset"] = {"id": DATASET_ID, "revision": DATASET_REVISION}
    # Convert Paths and other library argument values into JSON-safe strings.
    import json
    manifest = json.loads(json.dumps(manifest, default=str))
    write_json(output.with_suffix(".metadata.json"), manifest)
    print(f"Saved {output}. Evaluate it with evaluate.py before reporting results.")


if __name__ == "__main__":
    main()
