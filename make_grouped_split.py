"""Regroup source-name variants and exact copies into deterministic splits."""
import argparse
from collections import defaultdict
from pathlib import Path
import random
import shutil

from data_utils import CLASS_NAMES, audit_dataset, label_path, sha256, split_images, write_json


def make_grouped_split(data, output, seed=42):
    import yaml

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    images = [p for split in ("train", "val", "test") for p in split_images(data, split)]
    parents = list(range(len(images)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    by_name, by_hash = {}, {}
    for i, image in enumerate(images):
        for key, lookup in ((image.name.split(".rf.")[0], by_name), (sha256(image), by_hash)):
            if key in lookup:
                parents[find(i)] = find(lookup[key])
            else:
                lookup[key] = i
    groups = defaultdict(list)
    for i, image in enumerate(images):
        groups[find(i)].append(image)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    targets = {"train": .7 * len(images), "val": .2 * len(images), "test": .1 * len(images)}
    counts = dict.fromkeys(targets, 0)
    assignment = {}
    for group in shuffled:
        # Fill the most underrepresented split relative to its target size.
        split = min(targets, key=lambda s: counts[s] / targets[s])
        counts[split] += len(group)
        for image in group:
            destination = output / split / "images" / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            labels = output / split / "labels" / (image.stem + ".txt")
            labels.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f"Duplicate filename: {image.name}; inspect the source dataset")
            shutil.copy2(image, destination)
            shutil.copy2(label_path(image), labels)
            assignment[str(image.relative_to(image.parents[2]))] = split
    cfg = {"path": str(output), "train": "train/images", "val": "val/images",
           "test": "test/images", "nc": 4, "names": CLASS_NAMES}
    config = output / "data.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    audit = audit_dataset(config)
    write_json(output / "split_manifest.json", {
        "seed": seed, "source_yaml_sha256": sha256(data), "counts": counts,
        "groups": len(groups), "assignment": assignment, "audit": audit,
        "group_policy": "Same filename before .rf. OR identical image bytes; transitive closure",
        "limitation": "Source-name grouping cannot guarantee specimen-level independence. "
                      "Synthetic variants with different names may remain related. "
                      "This split is not the official RF100 benchmark split."})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="asbestos/data.local.yaml")
    parser.add_argument("--output", default="asbestos/grouped")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("Grouped data YAML:", make_grouped_split(args.data, args.output, args.seed))


if __name__ == "__main__":
    main()
