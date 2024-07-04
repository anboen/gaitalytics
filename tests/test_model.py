from pathlib import Path

import h5py
import pytest

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, C3dEventInputFileReader
from gaitalytics.model import DataCategory, Trial, trial_from_hdf5


@pytest.fixture()
def output_path(request):
    OUTPUT_HDF5_SMALL: Path = Path('out/test_small.hdf5')

    def delete_file():
        if OUTPUT_HDF5_SMALL.exists():
            try:
                OUTPUT_HDF5_SMALL.unlink()
            except PermissionError:
                pass

    delete_file()
    return OUTPUT_HDF5_SMALL


@pytest.fixture()
def input_path(request):
    INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')

    return INPUT_C3D_SMALL


class TestModel:

    def test_add(self, input_path):
        markers_input = MarkersInputFileReader(input_path)
        markers = markers_input.get_markers()
        new_markers = markers.copy(deep=True)
        labels = [f"{label}_new" for label in new_markers["channel"].values]
        new_markers = new_markers.assign_coords(channel=labels)

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.MARKERS, new_markers)

        rec_value = len(trial.get_data(DataCategory.MARKERS).coords["channel"])
        exp_value = 382

        assert rec_value == exp_value, f"Expected {exp_value} markers, got {rec_value}"

    def test(self, input_path):
        markers = MarkersInputFileReader(input_path).get_markers()
        analogs = AnalogsInputFileReader(input_path).get_analogs()
        events = C3dEventInputFileReader(input_path).get_events()

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.ANALOGS, analogs)
        trial.events = events

        rec_value = len(trial.get_all_data())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

    def test_save_empty_to_hdf5(self, output_path):
        trial = Trial()
        trial.to_hdf5(output_path)

    def test_save_to_existing_hdf5(self, input_path, output_path):
        markers = MarkersInputFileReader(input_path).get_markers()
        analogs = AnalogsInputFileReader(input_path).get_analogs()
        events = C3dEventInputFileReader(input_path).get_events()

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.ANALOGS, analogs)
        trial.events = events

        trial.to_hdf5(output_path)

        try:
            trial.to_hdf5(output_path)
            assert False, "Expected an exception when saving to an existing file"
        except FileExistsError:
            pass

    def test_save_to_hdf5(self, input_path, output_path):
        markers = MarkersInputFileReader(input_path).get_markers()
        analogs = AnalogsInputFileReader(input_path).get_analogs()
        events = C3dEventInputFileReader(input_path).get_events()

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.ANALOGS, analogs)
        trial.events = events

        trial.to_hdf5(output_path)

        assert output_path.exists(), f"Expected {output_path} to exist, but it does not"

        with h5py.File(output_path, 'r') as f:
            rec_value = len(f.keys())
            exp_value = 3
            assert rec_value == exp_value, f"Expected {exp_value} datasets, got {rec_value}"

    def test_load_hdf5(self, input_path, output_path):
        markers = MarkersInputFileReader(input_path).get_markers()
        analogs = AnalogsInputFileReader(input_path).get_analogs()
        events = C3dEventInputFileReader(input_path).get_events()

        trial = Trial()
        trial.add_data(DataCategory.MARKERS, markers)
        trial.add_data(DataCategory.ANALOGS, analogs)
        trial.events = events

        trial.to_hdf5(output_path)

        loaded_trial = trial_from_hdf5(output_path)

        rec_value = len(loaded_trial.get_all_data())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

        assert loaded_trial.events is not None, "Expected events to be loaded, but they are not"
        del loaded_trial
