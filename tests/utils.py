from pathlib import Path
import pytest

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
