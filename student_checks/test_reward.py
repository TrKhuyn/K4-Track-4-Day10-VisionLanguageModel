"""Student-owned reward tests; instructor's tests/ stays unchanged."""
import json
import pytest
from lab_reward import compute_reward, iou_reward

GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]
PERFECT = {"label": "pothole", "bbox_2d": [100, 100, 200, 200]}


@pytest.mark.parametrize("completion,gt,expected", [
    (json.dumps([PERFECT]), GT, 1.0),
    ("broken JSON", GT, 0.0),
    (json.dumps([{**PERFECT, "label": "crack"}]), GT, 0.2),
    (json.dumps([PERFECT, PERFECT]), GT, 0.6),
    ("[]", [], 1.0),
])
def test_five_required_cases(completion, gt, expected):
    assert compute_reward(completion, gt) == pytest.approx(expected)


def test_empty_prediction_on_damage_is_not_perfect():
    assert compute_reward("[]", GT) == pytest.approx(0.2)


def test_giant_box_gets_no_localization_reward():
    giant = {"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}
    assert compute_reward(json.dumps([giant]), GT) == pytest.approx(0.2)


def test_greedy_matching_uses_each_gt_once():
    assert compute_reward(json.dumps([PERFECT] * 3), GT) == pytest.approx(0.2 + 0.8 / 3)


def test_trl_conversational_completion_and_gt_alignment():
    result = iou_reward(
        completions=[[{"role": "assistant", "content": json.dumps([PERFECT])}],
                     [{"role": "assistant", "content": "[]"}]],
        gt_boxes=[GT, GT], prompts=[], trainer_state=None)
    assert result == pytest.approx([1.0, 0.2])


def test_group_alignment_cannot_silently_truncate():
    with pytest.raises(ValueError):
        iou_reward(["[]", "[]"], [[]])
