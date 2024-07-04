"""This module contains classes for segmenting the trial data with different methods.

The module provides classes for segmenting the trial data based on
gait events as well as a base class to implement additional methods.
"""

from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr

import gaitalytics.io as io
import gaitalytics.model as model


class _BaseSegmentation(ABC):
    @abstractmethod
    def segment(self, trial: model.Trial) -> model.BaseTrial:
        """Segments the trial data based on the segmentation method.

        Args:
            trial (model.Trial): The trial to be segmented.
        """
        raise NotImplementedError


class GaitEventsSegmentation(_BaseSegmentation):
    """A class for segmenting the trial data based on gait events.

    This class provides a method to segment the trial data based on gait events.
    It splits the trial data based on the event label and context.
    """

    def __init__(self, event_label: str = "Foot Strike"):
        """Initializes a new instance of the GaitEventsSegmentation class.

        Args:
            event_label (str): The label of the event to be used for segmentation.
        """
        self.event_label = event_label

    def segment(self, trial: model.Trial) -> model.SegmentedTrial:
        """Segments the trial data based on gait events and contexts.

        Args:
            trial (model.Trial): The trial to be segmented.

        Returns:
            model.SegmentedTrial: A new trial containing the segmented data.

        Raises:
            ValueError: If the trial does not have events.
        """
        events = trial.events
        if events is None:
            raise ValueError("Trial does not have events.")

        events_times = self._get_times_of_events(events)

        context_segments = model.SegmentedTrial()
        for context, times in events_times.items():
            cycle_segments = model.SegmentedTrial()
            for cycle_id in range(len(times) - 1):
                start_time = times[cycle_id]
                end_time = times[cycle_id + 1]
                cycle_segments.add_segment(
                    str(cycle_id),
                    self._get_segment(trial, start_time, end_time, cycle_id, context),
                )
            context_segments.add_segment(context, cycle_segments)

        return context_segments

    def _get_times_of_events(self, events: pd.DataFrame) -> dict[str, list]:
        """Gets the times of the events in the trial.

        This method splits the trial data based on the event label and context.

        Args:
            events (pd.DataFrame): The events in the trial.

        Returns:
            dict[str, list]: A dictionary containing the contexts
                as keys and the event times as values.
        """
        splits = {}
        interesting_events = events[
            events[io.EventInputFileReader.COLUMN_LABEL] == self.event_label
        ]
        contexts = events[io.EventInputFileReader.COLUMN_CONTEXT].unique()
        for context in contexts:
            context_events = interesting_events[
                interesting_events[io.EventInputFileReader.COLUMN_CONTEXT] == context
            ]
            splits[context] = context_events[io.EventInputFileReader.COLUMN_TIME].values
        return splits

    def _get_segment(
        self,
        trial: model.Trial,
        start_time: float,
        end_time: float,
        cycle_id: int,
        context: str,
    ) -> model.Trial:
        """Segments the trial data based on the start and end times.

        Args:
            trial (model.Trial): The trial to be segmented.
            start_time (float): The start time of the segment.
            end_time (float): The end time of the segment.
            cycle_id (int): The cycle id of the segment.
            context (str): The context of the segment.

        Returns:
            model.Trial: A new trial containing the segmented data.
        """
        trial_segment = model.Trial()
        # segment the data
        for category, data in trial.get_all_data().items():
            segment = data.sel(time=slice(start_time, end_time))
            self._update_attrs(segment, cycle_id, context)
            trial_segment.add_data(category, segment)
        # segment the events
        trial_segment.events = self._segment_events(trial.events, start_time, end_time)
        return trial_segment

    @staticmethod
    def _segment_events(
        events: pd.DataFrame | None, start_time: float, end_time: float
    ) -> pd.DataFrame:
        """Segments the events based on the start and end times.

        Args:
            events (pd.DataFrame): The events to be segmented.
            start_time (float): The start time of the segment.
            end_time (float): The end time of the segment.

        Returns:
            pd.DataFrame: A DataFrame containing the segmented events.
        """
        if events is None:
            raise ValueError("Events are not set.")

        return events[
            (events[io.EventInputFileReader.COLUMN_TIME] > start_time)
            & (events[io.EventInputFileReader.COLUMN_TIME] < end_time)
        ]

    @staticmethod
    def _update_attrs(segment: xr.DataArray, cycle_id: int, context: str):
        """Updates the attributes of the segment based on the data.

        Args:
            segment (xr.DataArray): The segment to be updated.
            cycle_id (int): The cycle id of the segment.
            context (str): The context of the segment.
        """
        start_frame = round(
            segment.coords["time"][0].data / (1 / segment.attrs["rate"]), 0
        )
        end_frame = round(
            segment.coords["time"][-1].data / (1 / segment.attrs["rate"]), 0
        )

        segment.attrs["start_frame"] = start_frame
        segment.attrs["end_frame"] = end_frame
        segment.attrs["cycle_id"] = cycle_id
        segment.attrs["context"] = context
