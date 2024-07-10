from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr

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
