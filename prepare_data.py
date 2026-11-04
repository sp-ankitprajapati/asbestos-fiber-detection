"""Download a fixed dataset revision and audit its annotations and splits."""
import argparse
from pathlib import Path

from data_utils import DATASET_ID, DATASET_REVISION, audit_dataset, write_json


def prepare_data(root="asbestos"):
    import yaml
    from huggingface_hub import snapshot_download

    root = Path(root).resolve()
    snapshot_download(repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
                      local_dir=str(root))
    source = root / "data.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["path"] = str(root)
    # Keep the upstream configuration intact; write a machine-local copy.
    local = root / "data.local.yaml"
    local.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="asbestos")
    parser.add_argument("--data", help="Audit an existing YAML without downloading")
    parser.add_argument("--output", default="reports/data_audit.json")
    args = parser.parse_args()
    data = Path(args.data) if args.data else prepare_data(args.root)
    report = audit_dataset(data)
    report.update(dataset=DATASET_ID, revision=DATASET_REVISION)
    write_json(args.output, report)
    for split, summary in report["splits"].items():
        print(split, summary)
    print("Cross-split exact duplicate groups:", len(report["cross_split_exact_duplicates"]))
    print("Cross-split source-name overlaps:", len(report["cross_split_source_name_overlaps"]))
    print("Data YAML:", data)
    print("Audit:", args.output)


if __name__ == "__main__":
    main()
