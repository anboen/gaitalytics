from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr

import gaitalytics.events as events
import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model


class BaseFeatureCalculation(ABC):
    def __init__(self, config: mapping.MappingConfigs, **kwargs):
        """Initializes a new instance of the BaseFeatureCalculation class.

        Args:
            config: The mapping configuration to use for the feature calculation.
        """
        self._config = config

    def calculate(self, trial: model.TrialCycles) -> xr.DataArray:
        """Calculate the features for a trial.

        Calls the _calculate method for each cycle in the trial and combines
        results into a single DataArray.

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        results: list = []

        context_dim: list[str] = []
        for context, context_cycles in trial.get_all_cycles().items():
            context_results: list = []
            cycle_dim: list[int] = []
            for cycle_id, cycle in context_cycles.items():
                feature = self._calculate(cycle)
                context_results.append(feature)
                cycle_dim.append(cycle_id)

            context_dim.append(context)
            context_results = xr.concat(
                context_results, pd.Index(cycle_dim, name="cycle")
            )
            results.append(context_results)

        result = xr.concat(results, pd.Index(context_dim, name="context"))
        return result

    @abstractmethod
    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the features for a trial.

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        raise NotImplementedError


class TimeSeriesFeatures(BaseFeatureCalculation):
    """Calculate time series features for a trial.

    This class calculates following time series features for a trial.
    - min
    - max
    - mean
    - median
    - std
    """

    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the time series features for a trial.

        Following features are calculated:
        - min
        - max
        - mean
        - median
        - std

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.
        """
        markers = trial.get_data(model.DataCategory.ANALYSIS)
        min_feat = markers.min(dim="time", skipna=True)
        max_feat = markers.max(dim="time", skipna=True)
        mean_feat = markers.mean(dim="time", skipna=True)
        median_feat = markers.median(dim="time", skipna=True)
        std_feat = markers.std(dim="time", skipna=True)
        features = xr.concat(
            [min_feat, max_feat, mean_feat, median_feat, std_feat],
            pd.Index(["min", "max", "mean", "median", "std"], name="feature"),
        )
        return features


class TemporalFeatures(BaseFeatureCalculation):
    def _calculate(self, trial: model.Trial) -> xr.DataArray:
        """Calculate the support times for a trial.

        Definitions of the temporal features
        Hollmann et al. 2011 (doi: 10.1016/j.gaitpost.2011.03.024)

        Args:
            trial: The trial for which to calculate the features.

        Returns:
            An xarray DataArray containing the calculated features.

        Raises:
            ValueError: If the sequence of events is incorrect.
        """
        trial_events = trial.events

        event_times = self.check_events_validity(trial_events)

        result_dict = self._calculate_supports(
            event_times[0], event_times[1], event_times[2], event_times[3]
        )
        result_dict["foot_off"] = event_times[2] / event_times[3]
        result_dict["opposite_foot_off"] = event_times[0] / event_times[3]
        result_dict["opposite_foot_contact"] = event_times[1] / event_times[3]
        result_dict["stride_time"] = event_times[3]
        result_dict["step_time"] = event_times[3] - event_times[1]
        result_dict["cadence"] = 60 / (event_times[3] / 2)
        xr_dict = {
            "coords": {
                "feature": {"dims": "feature", "data": list(result_dict.keys())}
            },
            "data": list(result_dict.values()),
            "dims": "feature",
        }
        return xr.DataArray.from_dict(xr_dict)

    @staticmethod
    def check_events_validity(
        trial_events: pd.DataFrame | None,
    ) -> tuple[float, float, float, float]:
        """Checks the sequence of events in the trial and returns the times.

        Args:
            trial_events: The events to be checked and extracted.

        Returns:
            The times of the events. in following order
            [contra_fo, contra_fs, ipsi_fo, end_time]

        Raises:
            ValueError: If the sequence of events is incorrect.
        """
        if trial_events is None:
            raise ValueError("Trial does not have events.")

        end_time = trial_events.attrs["end_time"]
        curren_context = trial_events.attrs["context"]
        cycle_id = trial_events.attrs["cycle_id"]

        if len(trial_events) < 3:
            raise ValueError(
                f"Missing events in segment {curren_context} nr. {cycle_id}"
            )
        ipsi_fo = trial_events[
            trial_events[io.EventInputFileReader.COLUMN_CONTEXT] == curren_context
        ]
        contra = trial_events[
            trial_events[io.EventInputFileReader.COLUMN_CONTEXT] != curren_context
        ]
        if len(ipsi_fo) != 1:
            raise ValueError(f"Error events sequence {curren_context} nr. {cycle_id}")
        if len(contra) != 2:
            raise ValueError(f"Error events sequence {curren_context} nr. {cycle_id}")

        contra_fs = contra[
            contra[io.EventInputFileReader.COLUMN_LABEL] == events.FOOT_STRIKE
        ]
        contra_fo = contra[
            contra[io.EventInputFileReader.COLUMN_LABEL] == events.FOOT_OFF
        ]

        ipsi_fo_time = ipsi_fo[io.EventInputFileReader.COLUMN_TIME].values[0]
        contra_fs_time = contra_fs[io.EventInputFileReader.COLUMN_TIME].values[0]
        contra_fo_time = contra_fo[io.EventInputFileReader.COLUMN_TIME].values[0]

        return contra_fo_time, contra_fs_time, ipsi_fo_time, end_time

    @staticmethod
    def _calculate_supports(
        contra_fo_time: float,
        contra_fs_time: float,
        ipsi_fo_time: float,
        end_time: float,
    ) -> dict[str, float]:
        """Calculate the support times for a trial.

        Args:
            contra_fo_time: The time of the contra foot off event.
            contra_fs_time: The time of the contra foot strike event.
            ipsi_fo_time: The time of the ipsi foot off event.
            end_time: The end time of the trial.

        Returns:
            The calculated support times.
        """
        double_support = (contra_fo_time + (ipsi_fo_time - contra_fs_time)) / end_time
        single_support = (contra_fs_time - contra_fo_time) / end_time
        return {"double_support": double_support, "single_support": single_support}
