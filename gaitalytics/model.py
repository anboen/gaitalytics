from enum import Enum
from pathlib import Path

import h5py
import pandas as pd
import xarray as xr


class DataCategory(Enum):
    """Enum class for the array categories."""
    MARKERS = "markers"
    ANALOGS = "analogs"


class Trial:

    def __init__(self):
        """
        Initializes a new instance of the Trial class.

        """
        self._data: dict[DataCategory, xr.DataArray] = {}
        self.events: pd.DataFrame | None = None

    def add_data(self, category: DataCategory, data: xr.DataArray):
        """
        Adds data to the trial.

        Args:
            category (DataCategory): The category of the data.
            data (xr.DataArray): The data array to be added.
        """
        if category in self._data:
            self._data[category] = xr.concat([self._data[category], data], dim="time")
        else:
            self._data[category] = data

    def get_data(self, category: DataCategory) -> xr.DataArray:
        """
        Gets the data from the trial.

        Args:
            category (DataCategory): The category of the data.

        Returns:
            xr.DataArray: The data array.
        """
        return self._data[category]

    def get_all_data(self) -> dict[DataCategory, xr.DataArray]:
        """
        Gets all data from the trial.

        Returns:
            dict[DataCategory, xr.DataArray]: A dictionary containing the data arrays.
        """
        return self._data

    def to_hdf5(self, file_path: Path):
        """
        Saves the trial data to an HDF5 file.

        Args:
            file_path (Path): The path to the HDF5 file.
        """
        if file_path.exists():
            raise FileExistsError(f"File {file_path} already exists.")

        for category, data in self.get_all_data().items():
            data.to_netcdf(file_path, group=category.value, mode="a")

        if self.events is not None:
            self.events.to_hdf(file_path, key="events")


def trial_from_hdf5(file_path: Path) -> Trial:
    """
    Loads trial data from an HDF5 file.

    Args:
        file_path (Path): The path to the HDF5 file.

    Returns:
        Trial: A new instance of the Trial class.
    """
    # Check if at least one of the entities groups is present in the file
    correct_file_format = False

    trial = Trial()

    with h5py.File(str(file_path), mode='r') as f:

        for category in DataCategory:
            if category.value in f.keys():
                with xr.load_dataarray(file_path, group=category.value) as group:
                    trial.add_data(category, group)
                correct_file_format = True

        if "events" in f.keys():
            trial.events = pd.read_hdf(file_path, key="events")
            correct_file_format = True

    if not correct_file_format:
        raise ValueError(f"File {file_path} does not have the correct format.")

    return trial
