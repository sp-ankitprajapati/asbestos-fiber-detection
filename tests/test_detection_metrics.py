"""Small cases with known counts, independent of any trained model."""
import unittest

from detection_metrics import box_iou, match_counts, summarize


class DetectionMetricsTests(unittest.TestCase):
    def test_iou_overlap_and_empty_area(self):
        self.assertAlmostEqual(box_iou((0, 0, 2, 2), (1, 1, 3, 3)), 1 / 7)
        self.assertEqual(box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0)

    def test_duplicate_detection_cannot_match_twice(self):
        targets = [(0, (0, 0, 1, 1))]
        predictions = [(0, .9, (0, 0, 1, 1)), (0, .6, (0, 0, 1, 1))]
        self.assertEqual(match_counts(predictions, targets, .5), (1, 1, 0))
        self.assertEqual(match_counts(predictions, targets, .8), (1, 0, 0))

    def test_wrong_class_and_missed_object(self):
        self.assertEqual(match_counts([(1, .9, (0, 0, 1, 1))],
                                      [(0, (0, 0, 1, 1))], .5), (0, 1, 1))

    def test_low_iou_is_false_positive_and_false_negative(self):
        self.assertEqual(match_counts([(0, .9, (0, 0, .2, .2))],
                                      [(0, (0, 0, 1, 1))], .5), (0, 1, 1))

    def test_empty_predictions_and_negative_image(self):
        self.assertEqual(match_counts([], [(0, (0, 0, 1, 1))], .5), (0, 0, 1))
        self.assertEqual(match_counts([(0, .9, (0, 0, 1, 1))], [], .5), (0, 1, 0))
        self.assertEqual(summarize(0, 0, 0)["f1"], 0)

    def test_confidence_boundary_is_inclusive(self):
        self.assertEqual(match_counts([(0, .5, (0, 0, 1, 1))],
                                      [(0, (0, 0, 1, 1))], .5), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
