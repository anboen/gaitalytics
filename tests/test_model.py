import shutil
from pathlib import Path

import h5py
import pytest
import xarray as xr

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, \
    C3dEventInputFileReader
from gaitalytics.model import DataCategory, Trial, SegmentedTrial, trial_from_hdf5
from gaitalytics.segmentation import GaitEventsSegmentation

INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
OUTPUT_PATH_SMALL: Path = Path('out/test_small')

INPUT_C3D_BIG: Path = Path('tests/data/test_big.c3d')
OUTPUT_PATH_BIG: Path = Path('out/test_big')


@pytest.fixture()
def trial_small(request):
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


@pytest.fixture()
def output_file_path_small(request):
    file_path = OUTPUT_PATH_SMALL.with_suffix('.hdf5')

    def delete_file():
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                pass

    delete_file()
    return file_path


@pytest.fixture()
def output_path_small(request):
    path = OUTPUT_PATH_SMALL

    def delete_file():
        if path.exists():
            try:
                shutil.rmtree(path)
            except PermissionError:
                pass

    delete_file()
    return path


@pytest.fixture()
def trial_big(request):
    markers = MarkersInputFileReader(INPUT_C3D_BIG).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_BIG).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_BIG).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events
    return trial


@pytest.fixture()
def output_file_path_big(request):
    file_path = OUTPUT_PATH_BIG.with_suffix('.hdf5')

    def delete_file():
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                pass

    delete_file()
    return file_path


@pytest.fixture()
def output_path_big(request):
    path = OUTPUT_PATH_BIG

    def delete_file():
        if path.exists():
            try:
                shutil.rmtree(path)
            except PermissionError:
                pass

    delete_file()
    return path


class TestTrial:

    def test_add(self):
        markers_input = MarkersInputFileReader(INPUT_C3D_SMALL)
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

    def test(self, trial_small):
        rec_value = len(trial_small.get_all_data())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

    def test_save_empty_to_hdf5(self, output_file_path_small):
        trial = Trial()
        with pytest.raises(ValueError):
            trial.to_hdf5(output_file_path_small)

        assert not output_file_path_small.exists(), f"Expected {output_file_path_small} to exist, but it does not"

    def test_save_to_existing_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)
        with pytest.raises(FileExistsError):
            trial_small.to_hdf5(output_file_path_small)

    def test_save_to_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)

        assert output_file_path_small.exists(), f"Expected {output_file_path_small} to exist, but it does not"

        with h5py.File(output_file_path_small, 'r') as f:
            rec_value = len(f.keys())
            exp_value = 3
            assert rec_value == exp_value, f"Expected {exp_value} datasets, got {rec_value}"

    def test_load_hdf5(self, trial_small, output_file_path_small):
        trial_small.to_hdf5(output_file_path_small)

        loaded_trial = trial_from_hdf5(output_file_path_small)

        rec_value = len(loaded_trial.get_all_data())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

        assert loaded_trial.events is not None, "Expected events to be loaded, but they are not"
        del loaded_trial

    def test_save_to_hdf5_big(self, trial_big, output_file_path_big):
        trial_big.to_hdf5(output_file_path_big)

        assert output_file_path_big.exists(), f"Expected {output_file_path_big} to exist, but it does not"

        with h5py.File(output_file_path_big, 'r') as f:
            rec_value = len(f.keys())
            exp_value = 3
            assert rec_value == exp_value, f"Expected {exp_value} datasets, got {rec_value}"

    def test_load_hdf5_big(self, trial_big, output_file_path_big):
        trial_big.to_hdf5(output_file_path_big)

        loaded_trial = trial_from_hdf5(output_file_path_big)

        rec_value = len(loaded_trial.get_all_data())
        exp_value = 2
        assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

        assert loaded_trial.events is not None, "Expected events to be loaded, but they are not"
        del loaded_trial

    def test_save_to_folder(self, trial_small, output_path_small):
        with pytest.raises(ValueError):
            trial_small.to_hdf5(output_path_small)


class TestSegmentedTrial:
    def test_empy(self, output_path_small):
        trial = SegmentedTrial()
        with pytest.raises(ValueError):
            trial.to_hdf5(output_path_small)

        assert not output_path_small.exists(), f"Expected {output_path_small} to exist, but it does not"

    def test_to_hdf5_small(self, trial_small, output_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        segments.to_hdf5(output_path_small)

        assert output_path_small.exists(), f"Expected {output_path_small} to exist, but it does not"

    def test_to_write_cycle(self, trial_small, output_file_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        trial: xr.DataArray = segments.get_segment("Left").get_segment("0").get_data(DataCategory.MARKERS)
        trial.to_netcdf(output_file_path_small, group="Left/0/markers")

        assert output_file_path_small.exists(), f"Expected {output_file_path_small} to exist, but it does not"

    def test_to_hdf5_big(self, trial_big, output_path_big):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_big)
        segments.to_hdf5(output_path_big)

        assert output_path_big.exists(), f"Expected {output_path_big} to exist, but it does not"

    def test_load_hdf5_small(self, trial_small, output_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        segments.to_hdf5(output_path_small)
        trial = trial_from_hdf5(output_path_small)
        assert output_path_small.exists(), f"Expected {output_path_small} to exist, but it does not"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            trial_from_hdf5(Path("foo.hdf5"))

    def test_save_segment_in_file(self, trial_small, output_file_path_small):
        segments = GaitEventsSegmentation("Foot Strike").segment(trial_small)
        with pytest.raises(ValueError):
            segments.to_hdf5(output_file_path_small)
