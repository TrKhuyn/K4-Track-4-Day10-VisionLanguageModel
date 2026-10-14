"""Verifiable IoU reward from GUIDE step 9; no ML dependencies."""
from scripts.score import iou_xyxy, parse_output


def compute_reward(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    labels = {box["label"] for box in gt_boxes} | {box["label"] for box in pred}
    for label in sorted(labels):
        gt = [box["bbox"] for box in gt_boxes if box["label"] == label]
        predictions = [box["bbox"] for box in pred if box["label"] == label]
        pairs = sorted(((iou_xyxy(a, b), i, j)
                        for i, a in enumerate(gt)
                        for j, b in enumerate(predictions)), reverse=True)
        used_gt, used_pred = set(), set()
        for value, i, j in pairs:
            if value < 0.5:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            total_iou += value
    return 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(pred), 1)


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part["text"] for part in content if part["type"] == "text")
    return "".join(parts)


def iou_reward(completions, gt_boxes, **kwargs):
    if len(completions) != len(gt_boxes):
        raise ValueError("Every completion must have its corresponding ground truth.")
    return [compute_reward(completion_text(text), gt)
            for text, gt in zip(completions, gt_boxes)]
