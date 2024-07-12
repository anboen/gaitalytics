from pathlib import Path

import pytest

import gaitalytics.mapping as mapping


class TestMappingConfigs:

    def test_load_analyse_markers(self):
        configs = mapping.MappingConfigs(Path('tests/config/pig_config.yaml'))
        rec_value = len(configs.get_markers_analysis())
        exp_value = 42
        assert rec_value == exp_value, f"Expected {exp_value} markers, got {rec_value}"

        rec_value = configs.get_markers_analysis()[0]
        exp_value = 'LHipAngles'
        assert rec_value == exp_value, f"Expected {exp_value} marker, got {rec_value}"

        rec_value = configs.get_analogs_analysis()
        assert not rec_value, "Expected no analogs"

    def test_load_analyse_analogs(self):
        configs = mapping.MappingConfigs(Path('tests/config/analogs_config.yaml'))
        rec_value = len(configs.get_analogs_analysis())
        exp_value = 1
        assert rec_value == exp_value, f"Expected {exp_value} analogs, got {rec_value}"

    def test_load_empy_config(self):
        configs = mapping.MappingConfigs(Path('tests/config/empty_config.yaml'))
        with pytest.raises(ValueError):
            configs.get_markers_analysis()

        with pytest.raises(ValueError):
            configs.get_analogs_analysis()

    def test_get_marker_mapping(self):
        configs = mapping.MappingConfigs(Path('tests/config/pig_config.yaml'))
        rec_value = configs.get_marker_mapping(mapping.MappedMarkers.L_TOE)
        exp_value = 'LTOE'
        assert rec_value == exp_value, f"Expected {exp_value} marker, got {rec_value}"
