"""Read the directory-based YOLO dataset used by this project."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["thick-dark-mark", "thick-light-mark", "thin-dark-mark", "thin-light-mark"]
DATASET_ID = "Libre-YOLO/asbestos"
DATASET_REVISION = "24d72f5484922bf95281cd9a6a9ef84760cd3d0c"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_model_names(model):
    names = model.names
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != CLASS_NAMES:
        raise ValueError("Checkpoint class names do not match the four fiber-mark labels. "
                         "Supply a detector trained on this dataset.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(yaml_path):
    import yaml

    yaml_path = Path(yaml_path).resolve()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if cfg.get("names") != CLASS_NAMES:
        raise ValueError(f"Expected these class names in this order: {CLASS_NAMES}")
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return cfg, root.resolve()


def split_images(yaml_path, split):
    cfg, root = load_dataset(yaml_path)
    location = cfg.get(split)
    if not isinstance(location, str):
        raise ValueError(f"{split} must name one image directory in data.yaml")
    folder = Path(location)
    if not folder.is_absolute():
        folder = root / folder
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found for {split}: {folder}")
    return images


def label_path(image_path):
    # This project supports the RF100 split/images and split/labels layout.
    image_path = Path(image_path)
    if image_path.parent.name != "images":
        raise ValueError(f"Expected a split/images directory: {image_path}")
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_labels(image_path):
    """Return (class_id, normalized xyxy). Empty files are valid negatives."""
    path = label_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(v) for v in line.split()]
            if len(values) != 5:
                raise ValueError("expected class x_center y_center width height")
            cls, x, y, w, h = values
            if cls != int(cls) or not 0 <= cls < len(CLASS_NAMES):
                raise ValueError("invalid class id")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("coordinates must be normalized and sizes positive")
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if min(box) < -1e-5 or max(box) > 1 + 1e-5:
                raise ValueError("box extends outside the image")
            labels.append((int(cls), box))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return labels


def audit_dataset(yaml_path):
    """Check annotation ranges, exact duplicates and source-name overlaps."""
    hashes, sources, splits = defaultdict(list), defaultdict(set), {}
    for split in ("train", "val", "test"):
        images = split_images(yaml_path, split)
        counts = [0] * len(CLASS_NAMES)
        empty = 0
        for image in images:
            labels = read_labels(image)
            empty += not labels
            for cls, _ in labels:
                counts[cls] += 1
            hashes[sha256(image)].append({"split": split, "file": image.name})
            # Roboflow suffixes differ across exports of a source image.
            sources[image.name.split(".rf.")[0]].add(split)
        splits[split] = {"images": len(images), "empty_labels": empty,
                         "instances": dict(zip(CLASS_NAMES, counts))}
    duplicate_groups = [group for group in hashes.values()
                        if len({item["split"] for item in group}) > 1]
    overlaps = {key: sorted(value) for key, value in sources.items() if len(value) > 1}
    return {"splits": splits, "cross_split_exact_duplicates": duplicate_groups,
            "cross_split_source_name_overlaps": overlaps,
            "note": "Name overlap is a leakage warning, not proof of identical images. "
                    "Hash checks do not detect resized or augmented copies."}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
