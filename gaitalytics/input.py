import collections
from abc import ABC, abstractmethod
from pathlib import PurePath, Path

import ezc3d
import numpy as np
import pandas as pd
import pyomeca


class _BaseInputFileReader(ABC):
    """
    Base class for input file readers.

    This class provides a common interface for reading input files.

    Attributes:
        file_path (PurePath): The path to the input file.

    """

    def __init__(self, file_path: PurePath):
        """
        Initializes a new instance of the _BaseInputFileReader class.

        Args:
            file_path (PurePath): The path to the input file.

        """
        self.file_path = file_path


class AnalogInputFileReader(_BaseInputFileReader):
    """
    Abstract base class for reading analog input files.
    """

    @abstractmethod
    def get_analog_size(self) -> int:
        """
        Get the size of the analog data.

        Returns:
            int: The size of the analog data.
        """
        pass

    @abstractmethod
    def get_analogs_frame_rate(self) -> float:
        """
        Get the frame rate of the analog data.

        Returns:
            float: The frame rate of the analog data.
        """
        pass

    @abstractmethod
    def get_analog_labels(self) -> list[str]:
        """
        Get the labels of the analog channels.

        Returns:
            list[str]: The labels of the analog channels.
        """
        pass

    @abstractmethod
    def get_analog_units(self) -> list[str]:
        """
        Get the units of the analog channels.

        Returns:
            list[str]: The units of the analog channels.
        """
        pass

    @abstractmethod
    def get_analog_data(self, index: int) -> pd.DataFrame:
        """
        Get the analog data for a specific index.

        Args:
            index (int): The index of the analog data.

        Returns:
            pd.DataFrame: The analog data for the specified index.
        """
        pass

    def get_analogs_times(self) -> np.ndarray:
        """
        Get the time values of the analog data.

        Returns:
            np.ndarray: The time values of the analog data.
        """
        pass


class PointInputFileReader(_BaseInputFileReader):
    """
    Abstract base class for reading point input files.

    This class defines the interface for reading point input files
    in gait analysis.
    Subclasses must implement the abstract methods to provide specific
    implementations.

    """

    @abstractmethod
    def get_points_size(self) -> int:
        """
        Get the number of points in the input file.

        Returns:
            int: The number of points.

        """
        pass

    @abstractmethod
    def get_points_frame_rate(self) -> float:
        """
        Get the frame rate of the input file.

        Returns:
            float: The frame rate.

        """
        pass

    @abstractmethod
    def get_points_labels(self) -> list[str]:
        """
        Get the labels of the points in the input file.

        Returns:
            list[str]: The labels of the points.

        """
        pass

    @abstractmethod
    def get_point_units(self) -> list[str]:
        """
        Get the units of the points.


        Returns:
            list[str]: The units of the points.

        """
        pass

    @abstractmethod
    def get_point_data(self, index: int) -> pd.DataFrame:
        """
        Get the data of the point at the specified index.

        Args:
            index (int): The index of the point.

        Returns:
            pd.Dataframe: The data of the point.

        """
        pass

    @abstractmethod
    def get_points_times(self) -> np.ndarray:
        """
        Get the time values of the points data.

        Returns:
            np.ndarray: The time values of the points data.

        """
        pass


class EventInputFileReader(_BaseInputFileReader):
    """
    Abstract base class for reading event input files.

    This class defines the interface for reading event input files
    in gait analysis.
    Subclasses must implement the abstract methods to provide specific
    implementations.

    """

    @abstractmethod
    def get_event_labels(self) -> list[str]:
        """
        Get the labels of the events in the input file.

        Returns:
            list[str]: The labels of the events.

        """
        pass

    @abstractmethod
    def get_event_times(self) -> np.ndarray:
        """
        Get the labels of the events in the input file.

        Returns:
            np.ndarray: An array of event times.
        """
        pass

    @abstractmethod
    def get_event_contexts(self) -> list[str]:
        """
        Get the contexts of the events in the input file.

        Returns:
            list[str]: The contexts of the events.
        """


class _EzC3dBaseReader(_BaseInputFileReader):
    """
    A base class for reading C3D files using the EzC3d library.

    Attributes:
        _c3d (EzC3d): The C3D object representing the file.

    """

    def __init__(self, file_path: PurePath):
        """
        Initializes a new instance of the _EzC3dBaseReader class.

        Args:
            file_path (PurePath): The path to the C3D file.

        """
        self._c3d = ezc3d.c3d(str(file_path))
        super().__init__(file_path)

    # help functions
    def print_structure(self):
        """
        Prints the structure of the C3D file.

        """
        self._print_structure(self._c3d)

    @staticmethod
    def _print_structure(node: any, prefix: str = ""):
        """
        Recursively prints the structure of a node in the C3D file.

        Args:
            node (any): The node to print the structure of.
            prefix (str): The prefix to add to each line.

        """
        for key in node.keys():
            sub_node = node.get(key)

            print(f"{prefix}{key}:")
            try:
                EzC3dFileHandler._print_structure(
                    sub_node, prefix=prefix + "  ")
            except AttributeError:
                pass
            finally:
                pass

    @staticmethod
    def create_time_array(last_frame, first_frame, frame_rate):
        """
        Create a time array based on the given parameters.

        Args:
            last_frame (int): The index of the last frame.
            first_frame (int): The index of the first frame.
            frame_rate (float): The frame rate in frames per second.

        Returns:
            numpy.ndarray: An array of time values.

        """
        # does not start from 0 therefore frame 1 is index 0
        last_index: int = last_frame + 1
        first_index: int = first_frame

        end_time = last_index / frame_rate
        start_time = first_index / frame_rate

        return np.arange(start_time, end_time, 1 / frame_rate)


class _EzC3dPointReader(_EzC3dBaseReader, PointInputFileReader, ABC):
    """
    A class for reading and extracting point data from a C3D file.

    This class provides methods to retrieve information about the points data
    in a C3D file, such as the size, frame rate, labels, units, and data of
    individual points.

    Attributes:
        _TYPE_LIST (list): List of valid point types.

    """

    _TYPE_LIST = ["ANGLE", "POWER", "FORCE", "MOMENT"]

    def get_points_size(self) -> int:
        """
        Gets the size of the points data in the C3D file.

        Returns:
            int: The size of the points data.

        """
        return self._c3d["header"]["points"]["size"]

    def get_points_frame_rate(self) -> float:
        """
        Gets the frame rate of the points data in the C3D file.

        Returns:
            float: The frame rate of the points data.

        """
        return self._c3d["header"]["points"]["frame_rate"]

    def _get_points_frist_frame(self) -> int:
        """
        Gets the first frame of the points data in the C3D file.

        Returns:
            int: The first frame of the points data.

        """
        return self._c3d["header"]["points"]["first_frame"]

    def _get_points_last_frame(self) -> int:
        """
        Gets the last frame of the points data in the C3D file.

        Returns:
            int: The last frame of the points data.

        """
        return self._c3d["header"]["points"]["last_frame"]

    def get_points_labels(self) -> list[str]:
        """
        Gets the labels of the points data in the C3D file.

        Returns:
            list: The labels of the points data.

        """
        return self._c3d["parameters"]["POINT"]["LABELS"]["value"]

    def _get_indicies_point_type(self, type: str) -> list[int]:
        """
        Returns a list of indices corresponding to the points of the given
        type.

        Args:
            type (str): The type of points to filter in c3d.

        Returns:
            list[str]: A list of indices corresponding to the points of the
                given type.
        """
        labels = self.get_points_labels()
        type_labels = self._c3d["parameters"]["POINT"][f"{type}S"]["value"]
        return [labels.index(type_label) for type_label in type_labels]

    def _get_point_type_unit(self, type: str) -> str:
        """
        Get the unit of a specific type in the C3D file.

        Args:
            type (str): The type of data to retrieve the unit for.

        Returns:
            str: The unit of the specified type.

        """
        return self._c3d["parameters"]["POINT"][f"{type}_UNITS"]["value"][0]

    def get_point_units(self) -> str:
        """
        Returns the units of measurement for each point in the C3D file.

        Returns:
            str: A list of units corresponding to each point in the C3D file.
        """
        units = self._c3d["parameters"]["POINT"]["UNITS"]["value"]
        point_size = self.get_points_size()
        if len(units) < point_size:
            unit = units[0]
            units = [unit for _ in range(point_size)]
            for type in self._TYPE_LIST:
                type_units = self._get_point_type_unit(type)
                type_indicies = self._get_indicies_point_type(type)
                for type_index in type_indicies:
                    units[type_index] = type_units
        return units

    def get_point_data(self, index: int) -> pd.DataFrame:
        """
        Gets the data of a point in the C3D file.

        Args:
            index (int): The index of the point to get the data of.

        Returns:
            pd.DataFrame: The data of the point.

        """
        base_label = self.get_points_labels()[index]
        data_dict: dict = {}
        for i, direction in enumerate(self._get_directions()):
            data = self._c3d["data"]["points"][i, index, :]
            label = f"{base_label}.{direction}"
            data_dict[label] = data
        return pd.DataFrame(data_dict)

    def _get_directions(self) -> list[str]:
        """
        Gets the index of a direction in the C3D file.

        Returns:
            list: The directions in the C3D file ordered by the index.
        """
        directions: dict[str, int] = {}
        for key in self._c3d["parameters"]["TRIAL"].keys():
            if key.endswith("_DIRECTION"):
                direction = key.replace("_DIRECTION", "")
                values = self._c3d["parameters"]["TRIAL"][key]["value"][0]
                directions[values] = direction
        directions = collections.OrderedDict(sorted(directions.items()))
        return list(directions.values())

    def get_points_times(self) -> np.ndarray:
        """
        Get the time values of the points data.

        Returns:
            np.ndarray: The time values of the points data.
        """
        return self.create_time_array(self._get_points_last_frame(),
                                      self._get_points_frist_frame(),
                                      self.get_points_frame_rate())


class _EzC3dAnalogReader(_EzC3dPointReader, AnalogInputFileReader, ABC):
    """
    A class for reading analog data from a C3D file.

    This class provides methods to retrieve information about the analog data,
    such as size, frame rate, labels, units, and the actual data.

    """

    def get_analog_size(self) -> int:
        """
        Gets the size of the analog data in the C3D file.

        Returns:
            int: The size of the analog data.

        """
        return self._c3d["header"]["analogs"]["size"]

    def get_analogs_frame_rate(self) -> float:
        """
        Gets the frame rate of the points data in the C3D file.

        Returns:
            float: The frame rate of the points data.

        """
        return self._c3d["header"]["analogs"]["frame_rate"]

    def _get_analogs_first_frame(self) -> int:
        """
        Gets the first frame of the points data in the C3D file.

        Returns:
            int: The first frame of the points data.

        """
        return self._c3d["header"]["analogs"]["first_frame"]

    def _get_analogs_last_frame(self) -> int:
        """
        Gets the last frame of the points data in the C3D file.

        Returns:
            int: The last frame of the points data.

        """
        return self._c3d["header"]["analogs"]["last_frame"]

    def get_analog_labels(self) -> list[str]:
        """
        Gets the labels of the analog data in the C3D file.

        Returns:
            list: The labels of the analog data.

        """
        return self._c3d["parameters"]["ANALOG"]["LABELS"]["value"]

    def get_analog_units(self) -> list[str]:
        """
        Gets the units of the analog data in the C3D file.

        Returns:
            list: The units of the analog data.

        """
        return self._c3d["parameters"]["ANALOG"]["UNITS"]["value"]

    def get_analog_data(self, index: int) -> pd.DataFrame:
        """
        Gets the data of an analog in the C3D file.

        Args:
            index (int): The index of the analog to get the data of.

        Returns:
            pd.DataFrame: The data of the analog.

        """
        label = self.get_analog_labels()[index]
        return pd.DataFrame({label: self._c3d["data"]["analogs"][0][index, :]})

    def get_analogs_times(self) -> np.ndarray:
        """
        Get the time values of the analog data.

        Returns:
            np.ndarray: The time values of the analog data.
        """
        return self.create_time_array(self._get_analogs_last_frame(),
                                      self._get_analogs_first_frame(),
                                      self.get_analogs_frame_rate())


class EzC3dFileHandler(_EzC3dAnalogReader, EventInputFileReader):
    """
    A class for handling C3D files in an easy and convenient way.
    Inherits from the EzC3dAnalogReader, EventInputFileReader and
    EventInputFileReader classes.
    """

    _MAX_EVENTS_PER_SECTION = 255

    def __init__(self, file_path: PurePath):
        """
        Initializes a new instance of the EzC3dFileHandler class.

        Args:
            file_path (PurePath): The path to the C3D file.

        """
        super().__init__(file_path)

    # events section

    def get_event_labels(self) -> list[str]:
        """
        Gets the labels of the events in the C3D file.

        Returns:
            list: The labels of the events.

        """
        section_base = "LABELS"
        labels = self._concat_sections(section_base)
        return labels

    def get_event_times(self) -> np.ndarray:
        """
        Returns the event times.

        Returns:
        np.ndarray: An array containing the event times.
        """
        section_base = "TIMES"
        return self._concat_sections(section_base)

    def get_event_contexts(self) -> list[str]:
        """
        Gets the contexts of the events in the C3D file.

        Returns:
            list: The contexts of the events.

        """
        section_base = "CONTEXTS"
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

    def _concat_sections(self, section_base):
        """
        Concatenates the values of the specified sections in the C3D file.

        Args:
            section_base (str): The base name of the sections to concatenate.

        Returns:
            list: A list containing the concatenated values of the sections.
        """
        values = []
        for section in self._get_sections(section_base):
            current_values = self._c3d["parameters"]["EVENT"][section]["value"]
            if type(current_values) is not list:
                # convert TIMES: c3d specifices values[0] as
                # minutes and values[1] as seconds
                current_values = current_values[1] + (current_values[0] * 60)
                current_values = current_values.tolist()
            values += current_values
        return values


class TrcPointInputFileReader(_BaseInputFileReader):

    def __init__(self, file_path: PurePath):
        self.analogs = pyomeca.Analogs.from_mot(Path(file_path))
        super().__init__(file_path)

    def get_size(self):
        return self.analogs.shape
