import h5py
import gaitalytics.model as model
from pathlib import Path
import pandas as pd


def write_to_hdf5(trial: model.Trial, path: Path):
    """
    Write a trial to an HDF5 file.

    Args:
        trial (Trial): The trial to write.
        path (str): The path to write the trial to.
    """
    with h5py.File(str(path), "a") as h5_file:
        _write_events(h5_file, trial.events)
        for data_category in trial.data_categories.values():
            _write_data_category(h5_file, data_category)


def _write_events(group: h5py.Group, events: model.Events):
    """
    Write events to the specified HDF5 group.

    Args:
        group (h5py.Group): The group to write the events to.
        events (Events): The events to write.
    """
    events_group = group.create_group("events")
    _write_pandas(events.table, events_group)


def _write_data_category(group: h5py.Group,
                         data_category: model.DataCategory):
    """
    Write data category to the HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to write the data category to.
        data_category (model.DataCategory): The data category to be written.
    """
    cat_group = group.create_group(data_category.type.value)
    cat_group.attrs["frame_rate"] = data_category.frame_rate
    cat_group.attrs["units"] = data_category.units
    _write_pandas(data_category.table, cat_group)


def _write_pandas(table: pd.DataFrame, group: h5py.Group):
    """
    Write a pandas DataFrame to an h5py Group.

    Args:
        table (pd.DataFrame): The pandas DataFrame to write.
        group (h5py.Group): The h5py Group to write the DataFrame to.
    """
    for column in table.columns:
        group.create_dataset(column, data=table[column].to_numpy())
