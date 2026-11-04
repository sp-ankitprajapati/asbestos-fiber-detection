import tempfile
import unittest
from pathlib import Path

from data_utils import CLASS_NAMES, audit_dataset, read_labels


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split in ("train", "valid", "test"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def image(self, split, name, label="0 .5 .5 .2 .2", content=b"example"):
        image = self.root / split / "images" / (name + ".jpg")
        image.write_bytes(content)
        (self.root / split / "labels" / (name + ".txt")).write_text(label)
        return image

    def test_invalid_class_and_nonfinite_coordinates(self):
        for label in ("4 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .5 .5 -1 .2"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                read_labels(self.image("train", "bad", label))

    def test_missing_label_is_not_silently_a_negative(self):
        image = self.image("train", "missing")
        (self.root / "train/labels/missing.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            read_labels(image)

    def test_empty_labels_are_valid(self):
        self.assertEqual(read_labels(self.image("train", "empty", "")), [])

    def test_duplicate_and_name_overlap_audit(self):
        import yaml

        self.image("train", "source.rf.aaa")
        self.image("valid", "source.rf.bbb")
        self.image("test", "independent", content=b"different")
        cfg = self.root / "data.yaml"
        cfg.write_text(yaml.safe_dump({"path": str(self.root), "train": "train/images",
                       "val": "valid/images", "test": "test/images", "names": CLASS_NAMES}))
        audit = audit_dataset(cfg)
        self.assertEqual(len(audit["cross_split_exact_duplicates"]), 1)
        self.assertEqual(audit["cross_split_source_name_overlaps"]["source"], ["train", "val"])
        self.assertEqual(audit["splits"]["train"]["instances"][CLASS_NAMES[0]], 1)

    def test_grouping_keeps_related_images_together(self):
        import json
        import yaml
        from make_grouped_split import make_grouped_split

        for split in ("train", "valid", "test"):
            for i in range(4):
                self.image(split, f"source{i}.rf.{split}", content=f"{split}-{i}".encode())
        cfg = self.root / "data.yaml"
        cfg.write_text(yaml.safe_dump({"path": str(self.root), "train": "train/images",
                       "val": "valid/images", "test": "test/images", "names": CLASS_NAMES}))
        first = make_grouped_split(cfg, self.root / "grouped_a")
        second = make_grouped_split(cfg, self.root / "grouped_b")
        a = json.loads((first.parent / "split_manifest.json").read_text())
        b = json.loads((second.parent / "split_manifest.json").read_text())
        self.assertEqual(a["assignment"], b["assignment"])
        self.assertFalse(a["audit"]["cross_split_source_name_overlaps"])
        self.assertEqual(sum(a["counts"].values()), 12)


if __name__ == "__main__":
    unittest.main()
