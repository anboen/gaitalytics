from abc import abstractmethod, ABC
from pathlib import PurePath, Path

import ezc3d
import numpy as np
import pandas as pd
import pyomeca
import xarray as xr


# Input Section
class _BaseInputFileReader(ABC):
    """
    Base class for input file readers.

    This class provides a common interface for reading input files.

    Attributes:
        file_path (PurePath): The path to the input file.

    """

    def __init__(self, file_path: Path):
        """
        Initializes a new instance of the _BaseInputFileReader class.

        Args:
            file_path (PurePath): The path to the input file.

        """
        self.file_path = file_path


class EventInputFileReader(_BaseInputFileReader):
    """
    Abstract base class for reading event input files.

    This class defines the interface for reading event input files
    in gait analysis.
    Subclasses must implement the abstract methods to provide specific
    implementations.

    """
    COLUMN_TIME = "time"
    COLUMN_LABEL = "label"
    COLUMN_CONTEXT = "context"
    COLUMN_ICON = "icon_id"

    @abstractmethod
    def get_events(self) -> pd.DataFrame:
        """
        Gets the events from the input file sorted by time.

        Returns:
            pd.DataFrame: A DataFrame containing the events.
        """
        pass


class C3dEventInputFileReader(EventInputFileReader):
    """
    A class for handling C3D files in an easy and convenient way.
    Inherits from the EzC3dAnalogReader, EventInputFileReader and
    EventInputFileReader classes.
    """
    _MAX_EVENTS_PER_SECTION = 255

    def __init__(self, file_path: Path):
        """
        Initializes a new instance of the EzC3dFileHandler class.

        Args:
            file_path (Path): The path to the C3D file.

        """
        self._c3d = ezc3d.c3d(str(file_path))
        super().__init__(file_path)

    def get_events(self) -> pd.DataFrame:
        """
       Gets the events from the input file sorted by time.

       Returns:
           pd.DataFrame: A DataFrame containing the events.
       """
        labels = self._get_event_labels()
        times = self._get_event_times()
        contexts = self._get_event_contexts()
        icons = self._get_event_icons()
        table = pd.DataFrame({
            self.COLUMN_TIME: times,
            self.COLUMN_LABEL: labels,
            self.COLUMN_CONTEXT: contexts,
            self.COLUMN_ICON: icons
        })
        table = table.sort_values(by=self.COLUMN_TIME, ascending=True)
        return table

    def _get_event_labels(self) -> list[str]:
        """
        Gets the labels of the events in the C3D file.

        Returns:
            list(str): The labels of the events.

        """
        section_base = "LABELS"
        labels = self._concat_sections(section_base)
        return labels

    def _get_event_times(self) -> list:
        """
        Returns the event times.

        Returns:
        np.ndarray: An array containing the event times.
        """
        section_base = "TIMES"
        return self._concat_sections(section_base)

    def _get_event_contexts(self) -> list[str]:
        """
        Gets the contexts of the events in the C3D file.

        Returns:
            list: The contexts of the events.

        """
        section_base = "CONTEXTS"
        return self._concat_sections(section_base)

    def _get_event_icons(self) -> list:
        """
        Gets the icons of the events in the C3D file.

        Returns:
            list: The icons of the events.
        """
        section_base = "ICON_IDS"
        return self._concat_sections(section_base)

    def _get_sections(self, section_base):
        """
        Gets the sections of the specified type in the C3D file.

        Args:
            section_base (str): The base name of the sections to get.

        Returns:
            list: A list containing the sections of the specified type.
        """
        sections = []
        for section in self._c3d["parameters"]["EVENT"].keys():
            if section.startswith(section_base):
                sections.append(section)
        return sections

    def _concat_sections(self, section_base) -> list:
        """
        Concatenates the values of the specified sections in the C3D file.

        Args:
            section_base (str): The base name of the sections to concatenate.

        Returns:
            list: A list containing the concatenated values of the sections.
        """
        values: list = []
        for section in self._get_sections(section_base):
            current_values = self._c3d["parameters"]["EVENT"][section]["value"]
            if len(current_values) == 2:
                # convert TIMES: c3d specifics values[0] as
                # minutes and values[1] as seconds
                current_values = current_values[1] + (current_values[0] * 60)
            if type(current_values) is np.ndarray:
                current_values = current_values.tolist()
            values += current_values
        return values


class _PyomecaInputFileReader(_BaseInputFileReader):
    def __init__(self, file_path: Path, pyomeca_class: type[pyomeca.Markers | pyomeca.Analogs]):
        """
        Initializes a new instance of the MarkersInputFileReader class.

        Args:
            file_path (Path): The path to the marker data file.

        """
        file_ext = file_path.name.split('.')[-1]
        if file_ext == 'c3d' and (pyomeca_class == pyomeca.Analogs or pyomeca_class == pyomeca.Markers):
            data = pyomeca_class.from_c3d(file_path)
        elif file_ext == 'trc' and pyomeca_class == pyomeca.Markers:
            raise NotImplementedError("TRC file format is not supported for markers")
        elif file_ext == 'mot' and pyomeca_class == pyomeca.Analogs:
            data = pyomeca_class.from_mot(file_path, pandas_kwargs={"sep": "\t", "index_col": False})
        elif file_ext == 'sto' and pyomeca_class == pyomeca.Analogs:
            raise NotImplementedError("STO file format is not supported for analogs")
        else:
            raise ValueError(f"Unsupported file extension: {file_ext} for class {pyomeca_class}")
        try:
            first_frame = data.attrs["first_frame"]
            frame_rate = data.attrs["rate"]
            data = self._to_absolute_time(data, first_frame, frame_rate)
        except KeyError:
            pass
        finally:
            self.data = data
        super().__init__(file_path)

    @staticmethod
    def _to_absolute_time(data: xr.DataArray, first_frame: int, rate: float) -> xr.DataArray:
        first_frame = first_frame
        data.coords["time"] = data.coords["time"] + (first_frame * 1 / rate)
        return data


class MarkersInputFileReader(_PyomecaInputFileReader):
    """
    A class for handling marker data in an easy and convenient way.
    Inherits from the Markers class in pyomeca and the _BaseInputFileReader
    class.
    """

    def __init__(self, file_path: Path):
        """
        Initializes a new instance of the MarkersInputFileReader class.

        Args:
            file_path (Path): The path to the marker data file.

        """
        super().__init__(file_path, pyomeca.Markers)

    def get_markers(self) -> xr.DataArray:
        """
        Gets the markers from the input file.

        Returns:
            xr.DataArray: An xarray DataArray containing the markers.
        """
        return self.data


class AnalogsInputFileReader(_PyomecaInputFileReader):
    """
    A class for handling analog data in an easy and convenient way.
    Inherits from the Analogs class in pyomeca and the _BaseInputFileReader
    class.
    """

    def __init__(self, file_path: Path):
        """
        Initializes a new instance of the AnalogsInputFileReader class.

        Args:
            file_path (Path): The path to the analog data file.

        """
        super().__init__(file_path, pyomeca.Analogs)

    def get_analogs(self) -> xr.DataArray:
        """
        Gets the analog data from the input file.

        Returns:
            xr.DataArray: An xarray DataArray containing the analog data.
        """
        return self.data

# Event to C3d Section
# TODO: Add c3d event writer
