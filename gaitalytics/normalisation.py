"""This module provides classes for batch normalisation of gait data in a trial."""

from abc import ABC, abstractmethod

import gaitalytics.model as model


class BaseNormaliser(ABC):
    """Base class for normalisers.

    This class provides a common interface for normalising data.
    """

    @abstractmethod
    def normalise(self, trial: model.BaseTrial) -> model.BaseTrial:
        """Normalises the input data.

        Args:
            trial (model.BaseTrial): The trial to be normalised.

        Returns:
            model.BaseTrial: A new trial containing the normalised data.
        """
        raise NotImplementedError


class LinearTimeNormaliser(BaseNormaliser):
    """A class for normalising data based on time.

    This class provides a method to normalise the data based on time.
    It scales the data to the range [0, 1] based on the time.
    """

    def __init__(self, n_frames: int = 100):
        """Initializes a new instance of the LinearTimeNormaliser class.

        Args:
            n_frames (int): The number of frames to normalise the data to.
        """
        self.n_frames: int = n_frames

    def normalise(self, trial: model.BaseTrial) -> model.BaseTrial:
        """Normalises the data based on time.

        Args:
            trial (model.BaseTrial): The trial to be normalised.

        Returns:
            model.BaseTrial: A new trial containing the normalised data.
        """
        if type(trial) is model.SegmentedTrial:
            trial = self._normalise_segmented_trial(trial)
        else:
            trial = self._normalise_trial(trial)
        return trial

    def _normalise_trial(self, trial: model.BaseTrial) -> model.Trial:
        new_trial = model.Trial()
        for data_category in trial.get_all_data():
            data = trial.get_data(data_category)
            norm_data = data.meca.time_normalize(n_frames=self.n_frames, norm_time=True)

            new_trial.add_data(data_category, norm_data)
        return new_trial

    def _normalise_segmented_trial(
        self, trial: model.BaseTrial
    ) -> model.SegmentedTrial:
        segments = model.SegmentedTrial()
        for key, segment in trial.get_all_segments().items():
            segments.add_segment(key, self.normalise(segment))

        return segments
