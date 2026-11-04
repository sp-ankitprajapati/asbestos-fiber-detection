"""Fixed-threshold detection counts with score-ordered, class-aware matching."""


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_counts(predictions, targets, confidence, match_iou=0.5):
    """Predictions: (class, score, xyxy); targets: (class, xyxy).

    Each target can match once. Process predictions by decreasing score and
    match the available same-class target with the largest IoU >= match_iou.
    Returns micro counts over all classes for one image, not AP.
    """
    remaining = set(range(len(targets)))
    tp = fp = 0
    for cls, score, box in sorted(predictions, key=lambda p: p[1], reverse=True):
        if score < confidence:
            continue
        candidates = [(box_iou(box, targets[i][1]), i)
                      for i in remaining if targets[i][0] == cls]
        overlap, index = max(candidates, default=(0.0, -1))
        if index >= 0 and overlap >= match_iou:
            tp += 1
            remaining.remove(index)
        else:
            fp += 1
    return tp, fp, len(remaining)


def summarize(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
