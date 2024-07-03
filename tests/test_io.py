import array as arr
from pathlib import Path

import pytest

from gaitalytics.io import C3dEventInputFileReader, MarkersInputFileReader, AnalogsInputFileReader

INPUT_C3D_SMALL: Path = Path('tests/data/test_small.c3d')
INPUT_TRC_SMALL: Path = Path('tests/data/test_small_mokka.trc')
INPUT_MOT_SMALL: Path = Path('tests/data/test_small.mot')

INPUT_C3D_BIG: Path = Path('tests/data/Baseline.5.c3d')


def test_c3d_events_small():
    c3d_events = C3dEventInputFileReader(INPUT_C3D_SMALL)
    events = c3d_events.get_events()
    assert len(events) == 13, f"Expected 13 events, got {len(events)}"
    assert events["time"].iloc[0] == pytest.approx(2.5), f"Expected 2.5, got {events['time'].iloc[0]}"
    assert events["label"].iloc[0] == "Foot Off", f"Expected Foot Off, got {events['label'].iloc[0]}"
    assert events["context"].iloc[0] == "Right", f"Expected Right, got {events['context'].iloc[0]}"
    assert events["icon_id"].iloc[0] == 2, f"Expected 2, got {events['context'].iloc[0]}"


def test_c3d_events_big():
    c3d_events = C3dEventInputFileReader(INPUT_C3D_BIG)
    events = c3d_events.get_events()
    assert len(events) == 865, f"Expected 865 events, got {len(events)}"

    # Test first event
    assert events["time"].iloc[0] == pytest.approx(1.03), f"Expected 1.03, got {events['time'].iloc[0]}"
    assert events["label"].iloc[0] == "Foot Off", f"Expected Foot Off, got {events['label'].iloc[0]}"
    assert events["context"].iloc[0] == "Left", f"Expected Left, got {events['context'].iloc[0]}"
    assert events["icon_id"].iloc[0] == 2, f"Expected 2, got {events['context'].iloc[0]}"

    # Test last event
    assert events["time"].iloc[-1] == pytest.approx(249.91), f"Expected 1.03, got {events['time'].iloc[-1]}"
    assert events["label"].iloc[-1] == "Foot Off", f"Expected Foot Off, got {events['label'].iloc[-1]}"
    assert events["context"].iloc[-1] == "Left", f"Expected Left, got {events['context'].iloc[-1]}"
    assert events["icon_id"].iloc[-1] == 2, f"Expected 2, got {events['context'].iloc[-1]}"


def test_c3d_markers_small():
    c3d_markers = MarkersInputFileReader(INPUT_C3D_SMALL)
    markers = c3d_markers.get_markers()
    assert len(markers.coords['channel']) == 191, f"Expected 191 markers, got {len(markers)}"
    exp_x_values = arr.array('f', [-1300.67626953, -1290.06420898, -1275.91687012, -1258.42175293, -1237.95727539])
    assert (markers.loc['x', 'RTOE'][0:5].data == exp_x_values).all(), \
        f"Expected {exp_x_values} , got {markers['x', 'RTOE', 0:5].data}"
    assert markers.coords['time'][0] == 2.48, \
        f"Expected {2.49}, got {markers.coords['time'].loc[0]}"


def test_c3d_markers_big():
    c3d_markers = MarkersInputFileReader(INPUT_C3D_BIG)
    markers = c3d_markers.get_markers()
    # Test markers length
    assert len(markers.coords['channel']) == 127, f"Expected 127 markers, got {len(markers)}"

    # Test first 5 and last 5 x values of LASIS
    exp_x_values = arr.array('f', [152.31216431, 152.31573486, 152.33493042, 152.36070251,
                                   152.38232422])
    rec_x_values = markers.loc['x', 'LASIS'][0:5].data
    assert (rec_x_values == exp_x_values).all(), f"Expected {exp_x_values}, got {rec_x_values}"

    # Test first 5 and last 5 x values of LASIS
    exp_x_values = arr.array('f', [165.27760315, 165.07537842, 164.69329834, 164.05558777,
                                   163.1633606])
    rec_x_values = markers.loc['x', 'LASIS'][-5:].data
    assert (rec_x_values == exp_x_values).all(), f"Expected {exp_x_values}, got {rec_x_values}"


def test_c3d_analog_small():
    c3d_analogs = AnalogsInputFileReader(INPUT_C3D_SMALL)
    analogs = c3d_analogs.get_analogs()

    # Test analogs length
    exp_value = 42
    rec_value = len(analogs.coords['channel'])
    assert rec_value == exp_value, f"Expected {exp_value} analogs, got {rec_value}"

    # Test first 5 values of Voltage.RERS
    exp_values = [0.024871826171875, 0.01682281494140625, 0.00705718994140625, -0.0067138671875, -0.01224517822265625]
    rec_values = list(analogs.loc['Voltage.RERS'][0:5].data)
    assert rec_values == exp_values, f"Expected {exp_values} , got {rec_values}"

    exp_value = 2.48
    rec_value = analogs.coords['time'][0]
    assert rec_value == exp_value, f"Expected {exp_value}, got {rec_value}"


def test_trc_markers_small():
    MarkersInputFileReader(INPUT_TRC_SMALL)


def test_mot_markers_small():
    mot_analogs = AnalogsInputFileReader(INPUT_MOT_SMALL)
    analogs = mot_analogs.get_analogs()
    # Test analogs length
    exp_value = 36
    rec_value = len(analogs.coords['channel'])
    assert rec_value == exp_value, f"Expected {exp_value} analogs, got {rec_value}"

    # Test first 5 values of Voltage.RERS
    exp_values = [-2.061715, -2.824317, -3.442063, -3.703248, -4.052287]
    rec_values = list(analogs.loc['ground_force4_vx'][0:5].data)
    assert rec_values == exp_values, f"Expected {exp_values} , got {rec_values}"

    exp_value = 2.48
    rec_value = analogs.coords['time'][0]
    assert rec_value == exp_value, f"Expected {exp_value}, got {rec_value}"


def test_wrong_file_format():
    try:
        MarkersInputFileReader(Path("foo.csv"))
        assert False, "Expected an unsupported file extension exception"
    except ValueError as e:
        pass
