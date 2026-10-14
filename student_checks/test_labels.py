"""Grounding label boundary/rounding checks; instructor tests stay unchanged."""
import json
import random
import pytest
from lab_runtime import yolo_to_box, objects_to_answer
from scripts.score import parse_output


@pytest.mark.parametrize('yolo', [
    [0.5, 0.5, 0.0002, 0.2],
    [0.5, 0.5, 0.2, 0.0002],
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0.5, 0, 0.2],
    [0.5, 1, 0.2, 0],
    [0.9999, 0.9999, 0.0001, 0.0001],
    [0.2, 0.3, 0.25, 0.1],
])
def test_rounding_and_clipping_preserve_valid_boxes(yolo):
    box = yolo_to_box('crack', yolo)
    x1, y1, x2, y2 = box['bbox_2d']
    assert all(type(x) is int and 0 <= x <= 1000 for x in box['bbox_2d'])
    assert x1 < x2 and y1 < y2
    assert parse_output(objects_to_answer([{'label':'crack','yolo':yolo}])) is not None


def test_ordinary_box_matches_the_guide_exactly():
    assert yolo_to_box('pothole', [.5,.5,.2,.2]) == {
        'label':'pothole','bbox_2d':[400,400,600,600]}


def test_real_vlcsnap_00058_zero_height_annotation():
    assert yolo_to_box('crack',[.639062,.786111,.003125,0]) == {
        'label':'crack','bbox_2d':[637,786,641,787]}


def test_answer_order_is_top_then_left_and_no_annotation_is_dropped():
    rows = [dict(label='crack',yolo=[.1,.8,.05,.01]),
            dict(label='pothole',yolo=[.7,.2,.1,.1]),
            dict(label='manhole',yolo=[.3,.2,.1,.1])]
    boxes=json.loads(objects_to_answer(rows))
    assert [box['label'] for box in boxes] == ['manhole','pothole','crack']
    assert len(boxes)==len(rows)
    assert objects_to_answer([])=='[]'


def test_many_valid_yolo_boxes_never_collapse_in_integer_space():
    rng=random.Random(42)
    for _ in range(1000):
        coords=[rng.random() for _ in range(4)]
        box=yolo_to_box('crack',coords)['bbox_2d']
        assert 0 <= box[0] < box[2] <= 1000
        assert 0 <= box[1] < box[3] <= 1000
