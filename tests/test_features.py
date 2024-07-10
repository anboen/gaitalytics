from pathlib import Path

import pytest

from gaitalytics.features import TimeSeriesFeatures
from gaitalytics.io import MarkersInputFileReader, C3dEventInputFileReader, \
    AnalogsInputFileReader, AnalysisInputReader
from gaitalytics.mapping import MappingConfigs
from gaitalytics.model import DataCategory, Trial
from gaitalytics.segmentation import GaitEventsSegmentation

INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
CONFIG_FILE = Path('tests/config/pig_config.yaml')


@pytest.fixture()
def configs(request):
    return MappingConfigs(CONFIG_FILE)


@pytest.fixture()
def trial_small(request):
    configs = MappingConfigs(CONFIG_FILE)
    markers = MarkersInputFileReader(INPUT_C3D_SMALL).get_markers()
    analogs = AnalogsInputFileReader(INPUT_C3D_SMALL).get_analogs()
    analysis = AnalysisInputReader(INPUT_C3D_SMALL, configs).get_analysis()
    events = C3dEventInputFileReader(INPUT_C3D_SMALL).get_events()

    trial = Trial()
    trial.add_data(DataCategory.MARKERS, markers)
    trial.add_data(DataCategory.ANALOGS, analogs)
    trial.add_data(DataCategory.ANALYSIS, analysis)
    trial.events = events

    return GaitEventsSegmentation().segment(trial)


class TestTimeSeriesFeatures:

    def test_calculation(self, configs, trial_small):
        features = TimeSeriesFeatures(configs).calculate(trial_small)

        for context in features.context.values:
            for marker in features.channel.values:
                for cycle in features.cycle.values:
                    if not features.loc[dict(context=context, channel=marker,
                                        cycle=cycle)].isnull().any():

                        min_value = features.loc[dict(context=context, channel=marker,
                                                      cycle=cycle, feature="min")]
                        max_value = features.loc[dict(context=context, channel=marker,
                                                      cycle=cycle, feature="max")]
                        mean_value = features.loc[dict(context=context, channel=marker,
                                                       cycle=cycle, feature="mean")]
                        median_value = features.loc[dict(context=context, channel=marker,
                                                         cycle=cycle, feature="median")]

                        assert min_value <= max_value, \
                            f"Min value is greater than max value for {context}, {marker}, {cycle}"
                        assert min_value <= mean_value, \
                            f"Min value is greater than max value for {context}, {marker}, {cycle}"
                        assert max_value >= mean_value, \
                            f"Min value is greater than max value for {context}, {marker}, {cycle}"
                        assert min_value <= median_value, \
                            f"Min value is greater than max value for {context}, {marker}, {cycle}"
                        assert max_value >= median_value, \
                            f"Min value is greater than max value for {context}, {marker}, {cycle}"
                    else:
                        if "Power" in marker or "Force" in marker or "Moment" in marker or "GRF" in marker:
                            assert True
                        else:
                            assert False, f"Missing values for {context}, {marker}, {cycle}"
