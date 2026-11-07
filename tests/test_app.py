"""Demo logic checked with a stand-in model, so Gradio and Ultralytics are not needed."""
import unittest
from types import SimpleNamespace

import numpy as np

from app import run_detection


class Values(list):
    """A list with tolist(), like the tensors Ultralytics returns."""

    def tolist(self):
        return list(self)


class FakeModel:
    """Records the image it receives and returns one fixed detection."""

    def predict(self, image, **settings):
        self.received = image.copy()
        boxes = SimpleNamespace(cls=Values([1]), conf=Values([0.87654]),
                                xyxy=Values([[1.04, 2.0, 3.0, 4.0]]))
        # Like Ultralytics, plot() returns the input image in its own (BGR) order.
        return [SimpleNamespace(names={1: "thick-light-mark"}, boxes=boxes,
                                plot=lambda: image.copy())]


class DemoTests(unittest.TestCase):
    def test_color_order_and_detection_rows(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 200  # pure red in RGB order
        model = FakeModel()
        annotated, rows = run_detection(model, rgb, confidence=0.25)
        self.assertEqual(model.received[0, 0].tolist(), [0, 0, 200])  # model gets BGR
        self.assertEqual(annotated[0, 0].tolist(), [200, 0, 0])  # display gets RGB
        self.assertEqual(rows, [["thick-light-mark", 0.8765, 1.0, 2.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
