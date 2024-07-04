from pathlib import Path

import pytest

from gaitalytics.events import SequenceEventChecker
from gaitalytics.io import C3dEventInputFileReader

INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')

INPUT_C3D_BIG: Path = Path('tests/data/test_big.c3d')
OUTPUT_PATH_BIG: Path = Path('out/test_big')


class TestEventSequenceChecker:

    def test_sequence_small(self):
        events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()
        checker = SequenceEventChecker()
        good, errors = checker.check_events(events)
        assert good, f"Event sequence is not correct but it should be. {errors}"

    def test_sequence_big(self):
        events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()
        checker = SequenceEventChecker()
        good, _ = checker.check_events(events)
        assert good, "Event sequence is not correct but it should be."

    def test_sequence_empty(self):
        checker = SequenceEventChecker()
        with pytest.raises(ValueError):
            checker.check_events(None)



