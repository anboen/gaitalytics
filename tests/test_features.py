from pathlib import Path

import pytest

from gaitalytics.features import TimeSeriesFeatures, TemporalFeatures
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
                        median_value = features.loc[
                            dict(context=context, channel=marker,
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


class TestTemporalFeatures:

    def test_calculation(self, configs, trial_small):
        features = TemporalFeatures(configs).calculate(trial_small)

        rec_value = features.shape[2]
        exp_value = 8
        assert rec_value == exp_value, f"Expected {exp_value} features, got {rec_value}"

        # Stride time
        rec_value = features.loc["Left", 0, "cadence"]
        exp_value = 113.2076
        assert rec_value == pytest.approx(
            exp_value, rel=1e-3), f"Expected {exp_value}, got {rec_value} in cadence for Left context"

        rec_value = features.loc["Right", 0, "cadence"]
        exp_value = 110.0917
        assert rec_value == pytest.approx(
            exp_value, rel=1e-3), f"Expected {exp_value}, got {rec_value} in cadence for Right context"

        # Stride time
        rec_value = features.loc["Left", 0, "stride_time"]
        exp_value = 1.06
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-2), f"Expected {exp_value}, got {rec_value} in stride_time for Left context"

        rec_value = features.loc["Right", 0, "stride_time"]
        exp_value = 1.09
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-2), f"Expected {exp_value}, got {rec_value} in stride_time for Right context"

        # Step time
        rec_value = features.loc["Left", 0, "step_time"]
        exp_value = 0.5
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-2), f"Expected {exp_value}, got {rec_value} in step_time for Left context"

        rec_value = features.loc["Right", 0, "step_time"]
        exp_value = 0.56
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-2), f"Expected {exp_value}, got {rec_value} in step_time for Right context"

        # Double support
        rec_value = features.loc["Left", 0, "double_support"]
        exp_value = 21.6981 / 100
        assert rec_value == pytest.approx(exp_value,
                                          rel=1e-5), f"Expected {exp_value}, got {rec_value} in double_support for Left context"

        rec_value = features.loc["Right", 0, "double_support"]
        exp_value = 23.8532 / 100
        assert rec_value == pytest.approx(exp_value,
                                          rel=1e-5), f"Expected {exp_value}, got {rec_value} in double_support for Right context"

        # Single support
        rec_value = features.loc["Left", 0, "single_support"]
        exp_value = 41.5094 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in single_support for Left context"

        rec_value = features.loc["Right", 0, "single_support"]
        exp_value = 35.7798 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in single_support for Right context"

        # Opposite foot off
        rec_value = features.loc["Left", 0, "opposite_foot_off"]
        exp_value = 11.3208 /100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in opposite_foot_off for Left context"

        rec_value = features.loc["Right", 0, "opposite_foot_off"]
        exp_value = 12.8440 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in opposite_foot_off for Right context"

        # Opposite foot contact
        rec_value = features.loc["Left", 0, "opposite_foot_contact"]
        exp_value = 52.8302 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in opposite_foot_contact for Left context"

        rec_value = features.loc["Right", 0, "opposite_foot_contact"]
        exp_value = 48.6239 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in opposite_foot_contact for Right context"

        # foot off
        rec_value = features.loc["Left", 0, "foot_off"]
        exp_value = 63.2075 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in foot_off for Left context"

        rec_value = features.loc["Right", 0, "foot_off"]
        exp_value = 59.6330 / 100
        assert rec_value == pytest.approx(
            exp_value,
            rel=1e-5), f"Expected {exp_value}, got {rec_value} in foot_off for Right context"
