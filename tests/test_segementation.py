from pathlib import Path

import pytest

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, \
    C3dEventInputFileReader
from gaitalytics.model import DataCategory, Trial, TrialCycles
from gaitalytics.segmentation import GaitEventsSegmentation


@pytest.fixture()
def small_trial(request):
    INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


@pytest.fixture()
def big_trial(request):
    INPUT_C3D_SMALL: Path = Path('tests/data/test_big.c3d')
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


class TestEventSegmentation:

    def test_empty_events(self, small_trial):
        small_trial.events = None
        with pytest.raises(ValueError):
            GaitEventsSegmentation("Foot Strike").segment(small_trial)

    def test_get_times_small(self, small_trial):
        segmentation = GaitEventsSegmentation("Foot Strike")
        contexts_events = segmentation._get_times_of_events(small_trial.events)

        assert len(
            contexts_events) == 2, f"Expected 2 contexts, got {len(contexts_events)}"

        rec_value = len(contexts_events["Left"])
        exp_value = 3
        assert rec_value == exp_value, f"Expected {exp_value} events for Left context, got {rec_value}"

        rec_value = len(contexts_events["Right"])
        exp_value = 3
        assert rec_value == exp_value, f"Expected {exp_value} events for Right context, got {rec_value}"

    def test_segment_small(self, small_trial):
        segmentation = GaitEventsSegmentation("Foot Strike")
        segments = segmentation.segment(small_trial)

        rec_value = len(segments.get_all_cycles())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} contexts, got {rec_value}"

        rec_value = len(segments.get_cycles_per_context("Left"))
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} segments for Left context, got {rec_value}"

        rec_value = len(segments.get_cycles_per_context("Right"))
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} segments for Right context, got {rec_value}"

        _test_start_end_frame(segments)

        _test_cycle_id_context(segments)

    def test_segment_big(self, big_trial):
        segmentation = GaitEventsSegmentation("Foot Strike")
        segments = segmentation.segment(big_trial)

        rec_value = len(segments.get_all_cycles())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} contexts, got {rec_value}"

        rec_value = len(segments.get_cycles_per_context("Left"))
        exp_value = 215
        assert rec_value == exp_value, f"Expected {exp_value} segments for Left context, got {rec_value}"

        rec_value = len(segments.get_cycles_per_context("Right"))
        exp_value = 215
        assert rec_value == exp_value, f"Expected {exp_value} segments for Right context, got {rec_value}"

        _test_start_end_frame(segments)

        _test_cycle_id_context(segments)

    def test_segmented_events_small(self, small_trial):
        segmentation = GaitEventsSegmentation("Foot Strike")
        segments = segmentation.segment(small_trial)
        for context, cycles in segments.get_all_cycles().items():
            for cycle_id, cycle in cycles.items():
                events = cycle.events
                rec_value = len(events)
                exp_value = 3
                message = f"Expected {exp_value} events, got {rec_value} in {context}: {cycle_id}"
                assert rec_value == exp_value, message


    def test_segmented_events_big(self, big_trial):
        segmentation = GaitEventsSegmentation("Foot Strike")
        segments = segmentation.segment(big_trial)
        for context, cycles in segments.get_all_cycles().items():
            for cycle_id, cycle in cycles.items():
                events = cycle.events
                rec_value = len(events)
                exp_value = 3
                message = f"Expected {exp_value} events, got {rec_value} in {context}: {cycle_id}"
                assert rec_value == exp_value, message



def _test_cycle_id_context(segments: TrialCycles):
    # Test cycle_id and context attrs
    for context, cycle_segments in segments.get_all_cycles().items():
        for cycle_id, cycle in cycle_segments.items():
            markers = cycle.get_data(DataCategory.MARKERS)
            analogs = cycle.get_data(DataCategory.ANALOGS)

            rec_value = markers.attrs["cycle_id"]
            exp_value = cycle_id
            message = f"Expected {exp_value} cycle_id, got {rec_value} in {context}: {cycle_id} markers"
            assert rec_value == exp_value, message

            rec_value = markers.attrs["context"]
            exp_value = context
            message = f"Expected {exp_value} context, got {rec_value} in {context}: {cycle_id} markers"
            assert rec_value == exp_value, message

            rec_value = analogs.attrs["cycle_id"]
            exp_value = cycle_id
            message = f"Expected {exp_value} cycle_id, got {rec_value} in {context}: {cycle_id} analogs"
            assert rec_value == exp_value, message

            rec_value = analogs.attrs["context"]
            exp_value = context
            message = f"Expected {exp_value} context, got {rec_value} in {context}: {cycle_id} analogs"
            assert rec_value == exp_value, message


def _test_start_end_frame(cycles: TrialCycles):
    # Test start end frame attrs
    # If events are not set based on low frame rate frames can overlap
    for context, cycle_segments in cycles.get_all_cycles().items():
        for cycle_idx in range(len(cycle_segments.keys()) - 1):
            current_key = list(cycle_segments.keys())[cycle_idx]
            next_key = list(cycle_segments.keys())[cycle_idx + 1]
            current_trial: Trial = cycle_segments[current_key]
            next_trial: Trial = cycle_segments[next_key]

            current_data = current_trial.get_data(DataCategory.MARKERS)
            next_data = next_trial.get_data(DataCategory.MARKERS)
            rec_value = abs(
                next_data.attrs["start_frame"] - current_data.attrs["end_frame"])
            exp_value = 1
            message = f"Expected max {exp_value} frame difference, got {rec_value} in {context}: {current_key} markers"
            assert rec_value <= exp_value, message

            current_data = current_trial.get_data(DataCategory.ANALOGS)
            next_data = next_trial.get_data(DataCategory.ANALOGS)
            rec_value = abs(
                next_data.attrs["start_frame"] - current_data.attrs["end_frame"])
            exp_value = 1
            message = f"Expected max {exp_value} frame difference, got {rec_value} in {context}: {current_key} analogs"
            assert rec_value <= exp_value, message
