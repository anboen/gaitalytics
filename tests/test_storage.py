import pytest
import gaitalytics.storage as storage
import gaitalytics.mapping as mapping
import gaitalytics.input as input
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


def test_write_trial():
    """
    Test the write_to_hdf5 function.
    """
    reader = input.EzC3dFileHandler(INPUT_FILE)
    trial = mapping.map_trial(reader, reader, reader)
    storage.write_to_hdf5(trial, OUTPUT_FILE)
    OUTPUT_FILE.unlink()


def test_heavy_write_trial():
    """
    Test the write_to_hdf5 function.
    """
    reader = input.EzC3dFileHandler(INPUT_FILE_LONG)
    trial = mapping.map_trial(reader, reader, reader)
    storage.write_to_hdf5(trial, OUTPUT_FILE_LONG)
    OUTPUT_FILE_LONG.unlink()
