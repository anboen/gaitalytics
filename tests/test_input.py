import pytest
from gaitalytics.input import EzC3dFileHandler
from pathlib import Path

INPUT_FILE: Path = Path('tests/data/test_healthy.c3d')
INPUT_FILE_LONG: Path = Path('tests/data/Baseline.5.c3d')
OUTPUT_FILE: Path = Path('out/test_healthy.hdf5')
OUTPUT_FILE_LONG: Path = Path('out/Baseline.5.hdf5')


def assert_equal(recieved, expected, name):
    message = f"{name} expected {expected}, got {recieved}"
    assert recieved == expected, message


def assert_equal_lists(recieved, expected, name):
    msg = f"{name} length diff expected {len(expected)}, got {len(recieved)}"
    assert len(recieved) == len(expected), msg

    for i in range(len(expected)):
        message = f"{name} index {i} expected {expected[i]}, got {recieved[i]}"
        assert recieved[i] == expected[i], message


def assert_approx_equal(recieved, expected, name):
    message = f"{name} expected {expected}, got {recieved}"
    assert recieved == pytest.approx(expected), message


def test_load_directions():
    c3d = EzC3dFileHandler(INPUT_FILE)
    assert_equal(c3d._get_directions(), ['X', 'Y', 'Z'], "directions")


def test_load_point_c3d():
    c3d = EzC3dFileHandler(INPUT_FILE)
    #  c3d.print_structure()
    assert_equal(c3d.get_points_size(), 191, "point size")
    assert_equal(c3d.get_points_frame_rate(), 100.0, "point frame_rate")
    assert_equal(c3d._get_indicies_point_type("ANGLE"),
                       [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                        122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                        133, 134, 135, 136], "type indicies ANGLE")
    assert_equal(c3d._get_indicies_point_type("POWER"),
                       [137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                        148, 149, 150, 151, 152], "type indicies POWER")
    assert_equal(c3d._get_indicies_point_type("FORCE"),
                       [153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                        164, 165, 166, 167, 168, 169, 170, 171, 172],
                       "type indicies FORCE")
    assert_equal(c3d._get_indicies_point_type("MOMENT"),
                       [173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                        184, 185, 186, 187, 188, 189, 190],
                       "type indicies MOMENT")
    assert_equal(c3d.get_point_units(), ['mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'mm', 'mm', 'mm', 'mm',
                                               'mm', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'deg',
                                               'deg', 'deg', 'deg', 'W', 'W',
                                               'W', 'W', 'W', 'W', 'W', 'W',
                                               'W', 'W', 'W', 'W', 'W', 'W',
                                               'W', 'W', 'N', 'N', 'N', 'N',
                                               'N', 'N', 'N', 'N', 'N', 'N',
                                               'N', 'N', 'N', 'N', 'N', 'N',
                                               'N', 'N', 'N', 'N', 'Nmm',
                                               'Nmm', 'Nmm', 'Nmm', 'Nmm',
                                               'Nmm', 'Nmm', 'Nmm', 'Nmm',
                                               'Nmm', 'Nmm', 'Nmm', 'Nmm',
                                               'Nmm', 'Nmm', 'Nmm', 'Nmm',
                                               'Nmm'], "point units")


def test_load_analog_c3d():
    c3d = EzC3dFileHandler(INPUT_FILE)
    assert_equal(c3d.get_analog_size(), 42, "analog size")
    assert_equal(c3d.get_analogs_frame_rate(),
                       1000.0, "analog frame_rate")
    print(c3d.get_analog_units())
    assert_equal(c3d.get_analog_units(),
                       ['N', 'N', 'N', 'Nmm', 'Nmm', 'Nmm', 'N', 'N', 'N',
                        'Nmm', 'Nmm', 'Nmm', 'N', 'N', 'N', 'Nmm', 'Nmm',
                        'Nmm', 'N', 'N', 'N', 'Nmm', 'Nmm', 'Nmm', 'V', 'V',
                        'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
                        'V', 'V', 'V', 'V', 'V'], "analog units")


def test_load_event_c3d():
    c3d = EzC3dFileHandler(INPUT_FILE)
    assert_equal(c3d.get_event_labels(),
                       ['Foot Strike', 'Foot Strike', 'Foot Strike',
                        'Foot Strike', 'Foot Strike', 'Foot Strike',
                        'Foot Off', 'Foot Off', 'Foot Off', 'Foot Off',
                        'Foot Off', 'Foot Off', 'Foot Off'], "event labels")
    assert_equal_lists(c3d.get_event_times(),
                             [3.4200000762939453, 4.480000019073486,
                              5.53000020980835, 2.890000104904175,
                              3.9800000190734863, 5.010000228881836,
                              4.090000152587891, 3.0299999713897705,
                              5.139999866485596, 3.5399999618530273,
                              4.599999904632568, 5.670000076293945, 2.5],
                             "event times")
    assert_equal(c3d.get_event_contexts(),
                       ['Left', 'Left', 'Left', 'Right', 'Right', 'Right',
                        'Left', 'Left', 'Left', 'Right', 'Right', 'Right',
                        'Right'], "event contexts")
    assert_equal(len(c3d.get_event_labels()), 13, "event size")
    assert_equal(len(c3d.get_event_times()), 13, "event size")
    assert_equal(len(c3d.get_event_contexts()), 13, "event size")


def test_load_heavy_event_c3d():
    c3d = EzC3dFileHandler(INPUT_FILE_LONG)
    assert_equal(len(c3d.get_event_labels()), 865, "event size")
    assert_equal(len(c3d.get_event_times()), 865, "event size")
    assert_equal(len(c3d.get_event_contexts()), 865, "event size")
    assert_equal_lists(c3d.get_event_times()[-6:],
                             [248.57999992370605,
                              248.77000045776367,
                              249.1599998474121,
                              249.3400001525879,
                              249.72999954223633,
                              249.9099998474121], "event times")
