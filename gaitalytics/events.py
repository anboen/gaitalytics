"""This module contains classes for checking and detecting events in a trial."""

from abc import ABC, abstractmethod

import pandas as pd

import gaitalytics.io as io


class BaseEventChecker(ABC):
    """Abstract class for event checkers.

    This class provides a common interface for checking events in a trial,
    which makes them interchangeable.
    """

    @abstractmethod
    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the events in the trial.

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the events are correct, False otherwise.
            list | None: A list of incorrect time slices,
                or None if the events are correct.
        """
        raise NotImplementedError


class SequenceEventChecker(BaseEventChecker):
    """A class for checking the sequence of events in a trial.

    This class provides a method to check the sequence of events in a trial.
    It checks the sequence of event labels and contexts.
    """

    _TIME_COLUMN = io.EventInputFileReader.COLUMN_TIME
    _LABEL_COLUMN = io.EventInputFileReader.COLUMN_LABEL
    _CONTEXT_COLUMN = io.EventInputFileReader.COLUMN_CONTEXT

    def check_events(self, events: pd.DataFrame) -> tuple[bool, list | None]:
        """Checks the sequence of events in the trial.

        In a normal gait cycle, the sequence of events is as follows:
        1. Foot Strike (right)
        2. Foot Off (left)
        3. Foot Strike (left)
        4. Foot Off (right)

        Args:
            events: The events to be checked.

        Returns:
            bool: True if the sequence is correct, False otherwise.
            list | None: A list time slice of incorrect sequence,
                or None if the sequence is correct.

        """
        if events is None:
            raise ValueError("Trial does not have events.")

        incorrect_times = []

        incorrect_labels = self._check_labels(events)
        incorrect_contexts = self._check_contexts(events)

        if incorrect_labels:
            incorrect_times.append(incorrect_labels)
        if incorrect_contexts:
            incorrect_times.append(incorrect_contexts)

        return not bool(incorrect_times), incorrect_times if incorrect_times else None

    def _check_labels(self, events: pd.DataFrame) -> list[tuple]:
        """Check alternating sequence of event labels.

        Expected sequence of event labels:
        1. Foot Strike
        2. Foot Off
        3. Foot Strike
        4. Foot Off

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        last_label = None
        last_time = None
        for i, label in enumerate(events[self._LABEL_COLUMN]):
            time = events[self._TIME_COLUMN].iloc[i]
            if label == last_label:
                incorrect_times.append((last_time, time))

            last_time = time
            last_label = label
        return incorrect_times

    def _check_contexts(self, events: pd.DataFrame) -> list[tuple]:
        """Check sequence of contexts of events.

        Expected sequence of event contexts:
        1. Right
        2. Right
        3. Left
        4. Left

        Args:
            events: The events to be checked.

        Returns:
            A list of incorrect time slices.
        """
        incorrect_times = []
        # Check the occurrence of the context in windows of 3 events.
        for i in range(len(events) - 3):
            max_occurance = (
                events[self._CONTEXT_COLUMN].iloc[i : i + 3].value_counts().max()
            )

            # If the context occurs more than twice in the window, it is incorrect.
            if max_occurance > 2:
                incorrect_times.append(
                    (
                        events[self._TIME_COLUMN].iloc[i],
                        events[self._TIME_COLUMN].iloc[i + 3],
                    )
                )

        return incorrect_times
