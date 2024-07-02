import pytest
import numpy as np
from gaitalytics import mapping
from gaitalytics.input import EzC3dFileHandler
from gaitalytics.model import DataCategoryType
from pathlib import Path

INPUT_FILE: Path = Path('tests/data/test_healthy.c3d')
INPUT_FILE_LONG: Path = Path('tests/data/Baseline.5.c3d')
OUTPUT_FILE: Path = Path('out/test_healthy.hdf5')
OUTPUT_FILE_LONG: Path = Path('out/Baseline.5.hdf5')


def assert_equal(recieved, expected, name):
    message = f"{name} expected {expected}, got {recieved}"
    assert recieved == expected, message


def assert_equal_lists(recieved, expected, name):
    msg = f"{name} length diff expected {len(expected)}, got {len(recieved)}"
    assert len(recieved) == len(expected), msg

    for i in range(len(expected)):
        message = f"{name} index {i} expected {expected[i]}, got {recieved[i]}"
        assert recieved[i] == expected[i], message


def assert_approx_equal(recieved, expected, name):
    message = f"{name} expected {expected}, got {recieved}"
    assert recieved == pytest.approx(expected), message


def test_map_point_category():
    c3d = EzC3dFileHandler(INPUT_FILE)
    category = mapping.map_point_data_category_input(c3d)
    assert_equal(category.type, DataCategoryType.POINT, "category type")
    assert_equal(category.frame_rate, 100.0, "frame rate")
    print(category.get_time_list())
    assert_approx_equal(
        np.mean(np.diff(category.get_time_list())), 0.01, "time list")


def test_map_analog_category():
    c3d = EzC3dFileHandler(INPUT_FILE)
    category = mapping.map_analog_data_category_input(c3d)
    assert_equal(category.type, DataCategoryType.ANALOG, "category type")
    assert_equal(category.frame_rate, 1000.0, "frame rate")
    assert_approx_equal(
        np.mean(np.diff(category.get_time_list())), 0.001, "time list")
    assert_equal(len(category.get_time_list()), 3370, "time list length")
    assert_equal(category.get_data_by_label('Force.Fx1').shape[0], 3370,
                       "data length")


def test_map_events():
    c3d = EzC3dFileHandler(INPUT_FILE)
    events = mapping.map_events(c3d)
    assert_equal_lists(events.get_event(
        0), ["Foot Strike", 3.4200000762939453, "Left"], "event 0")


def test_map_trial():
    c3d = EzC3dFileHandler(INPUT_FILE)
    trial = mapping.map_trial(c3d, c3d, c3d)
    assert_equal(len(trial.data_categories), 2, "categories length")
