from pathlib import Path

import h5py

from gaitalytics.io import MarkersInputFileReader, AnalogsInputFileReader, C3dEventInputFileReader
from gaitalytics.model import DataCategory, Trial, trial_from_hdf5

INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
OUTPUT_HDF5_SMALL: Path = Path('out/test_small.hdf5')


def test_add():
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


def test():
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events

    rec_value = len(trial.get_all_data())
    exp_value = 2
    assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"


def test_save_empty_to_hdf5():
    trial = Trial()
    trial.to_hdf5(OUTPUT_HDF5_SMALL)


def test_save_to_existing_hdf5():
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events

    trial.to_hdf5(OUTPUT_HDF5_SMALL)

    try:
        trial.to_hdf5(OUTPUT_HDF5_SMALL)
        assert False, "Expected an exception when saving to an existing file"
    except FileExistsError:
        pass

    OUTPUT_HDF5_SMALL.unlink()


def test_save_to_hdf5():
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events

    trial.to_hdf5(OUTPUT_HDF5_SMALL)

    with h5py.File(OUTPUT_HDF5_SMALL, 'r') as f:
        rec_value = len(f.keys())
        exp_value = 3
        assert rec_value == exp_value, f"Expected {exp_value} datasets, got {rec_value}"

    assert OUTPUT_HDF5_SMALL.exists(), f"Expected {OUTPUT_HDF5_SMALL} to exist, but it does not"

    OUTPUT_HDF5_SMALL.unlink()


def test_load_hdf5():
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.events = events

    trial.to_hdf5(OUTPUT_HDF5_SMALL)

    loaded_trial = trial_from_hdf5(OUTPUT_HDF5_SMALL)

    rec_value = len(loaded_trial.get_all_data())
    exp_value = 2
    assert rec_value == exp_value, f"Expected {exp_value} data categories, got {rec_value}"

    assert loaded_trial.events is not None, "Expected events to be loaded, but they are not"
