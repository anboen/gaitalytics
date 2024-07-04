"""This module provides classes for structuring, storing and loading trial data."""
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import h5py
import pandas as pd
import xarray as xr


class DataCategory(Enum):
    """Enum class for the array categories.

    This class provides the categories for the data arrays.


    Attributes:
        MARKERS (str): The markers category.
        ANALOGS (str): The analogs category.
    """

    MARKERS = "markers"
    ANALOGS = "analogs"


class BaseTrial(ABC):
    """Abstract base class for trials.

    This class provides a common interface for trials to load and save data.
    """
    @abstractmethod
    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Local implementation of the to_hdf5 method.

        Args:
            file_path (Path): The path to the HDF5 file.
            base_group (str): The base group to save the data.
                If None, the data will be saved in the root of the file.
                Default = None
        """
        raise NotImplementedError

    def to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Saves the trial data to an HDF5 file.

        Args:
            file_path (Path): The path to the HDF5 file.
            base_group (str): The base group to save the data.
                If None, the data will be saved in the root of the file.
                Default = None

        Raises:
            FileExistsError: If the file already exists.
            ValueError: If the trial is a segmented trial and
                the file path is a single file.
            ValueError: If the trial is a trial and the file path is a folder.
        """
        if file_path.exists():
            raise FileExistsError(f"File {file_path} already exists.")
        elif type(self) is SegmentedTrial and file_path.suffix:
            raise ValueError("Cannot save a segmented trial in a single file.")
        elif type(self) is Trial and not file_path.suffix:
            raise ValueError("Cannot save a trial in folder")

        paths, data, groups = self._to_hdf5(file_path, base_group)
        if len(data) > 0:
            xr.save_mfdataset(data, paths, groups=groups, mode="a")
        else:
            raise ValueError("No data to save.")


class Trial(BaseTrial):
    """Represents a trial.

    A trial is a collection of data arrays (typically markers & analogs) and events.
    """
    def __init__(self):
        """Initializes a new instance of the Trial class."""
        self._data: dict[DataCategory, xr.DataArray] = {}
        self._events: pd.DataFrame | None = None

    @property
    def events(self) -> pd.DataFrame | None:
        """Gets the events in the trial.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the events if present.
        """
        return self._events

    @events.setter
    def events(self, events: pd.DataFrame):
        """Sets the events in the trial.

        Args:
            events (pd.DataFrame): The events to be set.
        """
        self._events = events

    def add_data(self, category: DataCategory, data: xr.DataArray):
        """Adds data to the trial.

        Args:
            category (DataCategory): The category of the data.
            data (xr.DataArray): The data array to be added.
        """
        if category in self._data:
            self._data[category] = xr.concat([self._data[category], data], dim="time")
        else:
            self._data[category] = data

    def get_data(self, category: DataCategory) -> xr.DataArray:
        """Gets the data from the trial.

        Args:
            category (DataCategory): The category of the data.

        Returns:
            xr.DataArray: The data array.
        """
        return self._data[category]

    def get_all_data(self) -> dict[DataCategory, xr.DataArray]:
        """Gets all data from the trial.

        Returns:
            dict[DataCategory, xr.DataArray]: A dictionary containing the data arrays.
        """
        return self._data

    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Saves trial into an HDF5 file.

        Structure:

        - root
            - markers
                - xarray.DataArray
            - analogs
                - xarray.DataArray
            - events
                - xarray.Dataset

        Args:
            file_path (Path): The path to the HDF5 file.
            base_group (str): The base group to save the data.
                If None, the data will be saved in the root of the file.
                Default = ""
        """
        if base_group is None:
            base_group = ""
        else:
            base_group = f"{base_group}"

        groups = []
        data = []
        paths = []

        if self.get_all_data() is not None and len(self.get_all_data()) > 0:
            groups += [
                f"{base_group}{category.value}"
                for category in self.get_all_data().keys()
            ]
            data += [data.to_dataset() for data in self.get_all_data().values()]
            paths += [file_path for _ in groups]

        if self.events is not None:
            groups.append(f"{base_group}events")
            data.append(self.events.to_xarray())
            paths.append(file_path)

        return paths, data, groups


class SegmentedTrial(BaseTrial):
    """Represents a segmented trial."""

    def __init__(self):
        """Initializes a new instance of the SegmentedTrial class."""
        self._segments: dict[str, BaseTrial] = {}

    def add_segment(self, key: str, segment: BaseTrial):
        """Adds a segment to the segmented trial.

        Args:
            key (str): The key of the segment.
            segment (BaseTrial): The segment to be added.
        """
        self._segments[key] = segment

    def get_segment(self, key: str) -> BaseTrial:
        """Gets a segment from the segmented trial.

        Args:
            key (str): The key of the segment.

        Returns:
            BaseTrial: The segment.
        """
        return self._segments[key]

    def get_all_segments(self) -> dict[str, BaseTrial]:
        """Gets all segments from the segmented trial.

        Returns:
            dict[str, BaseTrial]: A dictionary containing the segments.
        """
        return self._segments

    def _to_hdf5(self, file_path: Path, base_group: str | None = None):
        """Recursively saves the segmented trial data to an HDF5 file.

        Unfortunately, writing a huge a mount of separate arrays in a single file
        is not efficient at the moment.
        So the data is saved in separate files.

        Structure example of GaitEventsSegmentation:
        /folder
            - 0.h5 (cycle_id)
                - Left
                    - markers
                        - xarray.DataArray
                    - analogs
                        - xarray.DataArray
                    - events
                        - xarray.DataSet
                - Right
                    - markers
                        - xarray.DataArray
                    - analogs
                        - xarray.DataArray
                    - events
                        - xarray.DataSet
            - ...

        Args:
            file_path (Path): The path to the HDF5 file.
            base_group (str): The base group to save the data.
                If None, the data will be saved in the root of the file.
                Default = None
        """
        if base_group is None:
            base_group = ""
        else:
            base_group = f"{base_group}/"

        groups = []
        data = []
        paths = []

        for key, segment in self.get_all_segments().items():
            if file_path.name.split(".")[-1] != "h5":
                file_path.mkdir(parents=True, exist_ok=True)

            new_file = file_path
            new_base_group = base_group
            if type(segment) is Trial:
                new_file = (new_file / key).with_suffix(".h5")
            elif type(segment) is SegmentedTrial:
                new_base_group = f"{base_group}{key}"

            seg_groups, seg_data, seg_path = segment._to_hdf5(
                new_file, base_group=f"{new_base_group}"
            )
            groups += seg_groups
            data += seg_data
            paths += seg_path

        return paths, data, groups


def trial_from_hdf5(file_path: Path) -> Trial:
    """Loads trial data from an HDF5 file.

    Args:
        file_path (Path): The path to the HDF5 file.

    Returns:
        Trial: A new instance of the Trial class.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    elif file_path.suffix:
        trial = _load_trial_file(file_path)
    else:
        trial = _load_segmented_trial_file(file_path)

    return trial


def _load_segmented_trial_file(file_path: Path) -> SegmentedTrial:
    context_segments = SegmentedTrial()

    for file in file_path.glob("**/*.h5"):
        with h5py.File(str(file), "r") as f:
            cycle_id = file.name.replace(".h5", "")
            for context in f.keys():
                if context not in context_segments.get_all_segments():
                    context_segments.add_segment(context, SegmentedTrial())

                trial = _load_trial(False, f[context], file)
                context_segments.get_segment(context).add_segment(cycle_id, trial)
    return context_segments


def _load_trial_file(file_path):
    """Loads a trial from an HDF5 file.

    Args:
        file_path (Path): The path to the HDF5 file.
    """
    # Check if at least one of the entities groups is present in the file
    correct_file_format = False

    with h5py.File(str(file_path), "r") as f:
        trial = _load_trial(correct_file_format, f, file_path)

    return trial


def _load_trial(correct_file_format, group, file_path):
    trial = Trial()
    for category in DataCategory:
        if category.value in group.keys():
            with xr.load_dataarray(
                    file_path, group=f"{group.name}/{category.value}"
            ) as data:
                trial.add_data(category, data)

            correct_file_format = True
    if "events" in group.keys():
        with xr.load_dataset(file_path, group=f"{group.name}/events") as events:
            trial.events = events.to_dataframe()
        correct_file_format = True

    if not correct_file_format:
        raise ValueError(f"File {file_path} does not have the correct format.")

    return trial
