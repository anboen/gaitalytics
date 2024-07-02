from enum import Enum
import numpy as np
import pandas as pd


class DataCategoryType(Enum):
    """
    Enum class representing the types of data categories.
    """
    POINT = "points"
    ANALOG = "analogs"


class DataCategory:
    """
    Class representing a data category with type, data, and frame rate.
    """
    COLUMN_NAME_TIME = "time"

    def __init__(self,
                 type: DataCategoryType,
                 frame_rate: float,
                 time_list: list[float] = None
                 ):
        """
        Initialize a DataCategory object.

        Args:
            type (DataCategoryType): The type of the data category.
             data (dict[str, DataEntity]): The data entities associated
             with the category.
            frame_rate (float): The frame rate of the data.
        """
        self.type: DataCategory = type
        self.frame_rate: float = frame_rate
        self.table: pd.DataFrame = None
        self.units: list[str] = []
        if time_list is not None:
            self.table: pd.DataFrame = pd.DataFrame(
                {self.COLUMN_NAME_TIME: time_list})
        else:
            self.table: pd.DataFrame = None

    def get_time_list(self) -> np.ndarray:
        """
        Get the time list of the data category.

        Returns:
            np.ndarray: The time list of the data category.
        """
        return self.table[self.COLUMN_NAME_TIME].to_numpy()

    def get_data_by_label(self, label: str) -> np.ndarray:
        """
        Get the data associated with a specific label.

        Args:
            label (str): The label of the data entity.

        Returns:
            np.ndarray: The data associated with the label.
        """
        return self.table[label].to_numpy()

    def add_data_entity(self,
                        label: str,
                        data: pd.DataFrame,
                        unit: str):
        """
        Add a data entity to the data category.

        Args:
            label (str): The label of the data entity.
            data (pd.DataFrame): The data of the data entity.
            unit (str): The unit of the data entity.
        """

        self.table = pd.concat([self.table, data], axis=1)
        self.units.append(unit)


class Events:
    """
    Class representing an event.
    """
    COLUMN_NAME_LABLE = "label"
    COLUMN_NAME_TIME = "time"
    COLUMN_NAME_CONTEXT = "context"

    def __init__(self,
                 label: list[str],
                 time: list[float],
                 context: list[str]):
        """
        Initialize an Events object.

        Args:
            label (list[str]): The label of the event.
            time (list[float]): The time of the event.
            context (list[str]): The context of the event.
        """
        self.table = pd.DataFrame({
            self.COLUMN_NAME_LABLE: label,
            self.COLUMN_NAME_TIME: time,
            self.COLUMN_NAME_CONTEXT: context
        })

    def get_event(self, index: int) -> list[str, float, str]:
        """
        Get the event at a specific index.

        Args:
            index (int): The index of the event.

        Returns:
            list[str, float, str]: The event at the index.
                (label, time, context)
        """
        return self.table[self.COLUMN_NAME_LABLE][index], \
            self.table[self.COLUMN_NAME_TIME][index], \
            self.table[self.COLUMN_NAME_CONTEXT][index]


class Trial:
    """
    Class representing a trial.
    """

    def __init__(self):
        """
        Initialize a Trial object.
        """
        self.data_categories: dict[DataCategoryType, DataCategory] = {}
        self.events: Events = None

    def add_data_category(self, data_category: DataCategory):
        """
        Add a add Data Category to the trial.

        Args:
            data_category (DataCategory): The data category to add.
        """
        self.data_categories[data_category.type] = data_category
