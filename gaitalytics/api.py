from functools import wraps
from inspect import signature, Parameter
from pathlib import Path

import pandas as pd

import gaitalytics.events as events
import gaitalytics.io as io
import gaitalytics.mapping as mapping
import gaitalytics.model as model
import gaitalytics.normalisation as normalisation
import gaitalytics.segmentation as segmentation


class _PathConverter:
    """A decorator to convert Path | str annotations to Path objects.

    This decorator is used to convert parameters with Path | str annotations
    to Path objects.
    """

    def __init__(self, func):
        wraps(func)(self)
        self.func = func
        self.sig = signature(func)

    def __call__(self, *args, **kwargs):
        bound_arguments = self.sig.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        new_args = []
        for name, value in bound_arguments.arguments.items():
            param: Parameter = self.sig.parameters[name]
            # Check if the annotation is Path | str
            if param.annotation == Path | str or param.annotation == Path | str | None:
                if isinstance(value, (Path, str)):
                    value = Path(value)
            if param.kind in [
                param.VAR_POSITIONAL,
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD,
            ]:
                new_args.append(value)
            else:
                bound_arguments.arguments[name] = value

        return self.func(*new_args, **bound_arguments.kwargs)


@_PathConverter
def load_config(config_path: Path | str) -> mapping.MappingConfigs:
    """Loads the mapping configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A MappingConfigs object.
    """
    return mapping.MappingConfigs(config_path)


@_PathConverter
def load_c3d_trial(
    c3d_file: Path | str, configs: mapping.MappingConfigs
) -> model.Trial:
    """Loads a Trial from a c3d file.

    Be aware that all the required data for the trial must be present in the c3d file.
    i.e. markers, analogs, events, etc.

    Args:
        c3d_file: The path to the c3d file.
        configs: The mapping configurations

    Returns:
        A Trial object.
    """
    markers = io.MarkersInputFileReader(c3d_file).get_markers()
    analogs = io.AnalogsInputFileReader(c3d_file).get_analogs()
    analysis = io.AnalysisInputReader(c3d_file, configs).get_analysis()
    event_table = io.C3dEventInputFileReader(c3d_file).get_events()

    trial = model.Trial()
    trial.add_data(model.DataCategory.MARKERS, markers)
    trial.add_data(model.DataCategory.ANALOGS, analogs)
    trial.add_data(model.DataCategory.ANALYSIS, analysis)
    trial.events = event_table

    return trial


def detect_events(
    trial: model.Trial, config: mapping.MappingConfigs, method: str = "Marker", **kwargs
) -> pd.DataFrame:
    """Detects the events in the trial.

    Args:
        trial: The trial to detect the events for.
        config: The mapping configurations
        method: The method to use for detecting the events.
        Currently, only "Marker" is supported, which implements
        the method from Zenis et al. 2006.
        Default is "Marker".

    Returns:
        A DataFrame containing the detected events.
    """

    match method:
        case "Marker":
            method_obj = events.MarkerEventDetection(config, **kwargs)
        case _:
            raise ValueError(f"Unsupported method: {method}")

    event_table = method_obj.detect_events(trial)
    return event_table


def check_events(event_table: pd.DataFrame, method: str = "sequence"):
    """Checks the events in the trial.

    Args:
        event_table: The event table to check.
        method: The method to use for checking the events.
        Currently, only supports "sequence" which checks the sequence of events
        in terms of context and label. Default is "sequence".

    Returns:
        The trial with the checked events.
    """
    match method:
        case "sequence":
            checker = events.SequenceEventChecker()
        case _:
            raise ValueError(f"Unsupported method: {method}")
    good, errors = checker.check_events(event_table)
    if not good:
        raise ValueError(f"Event sequence is not correct: {errors}")


@_PathConverter
def write_events_to_c3d(
    c3d_path: Path | str,
    event_table: pd.DataFrame,
    output_path: Path | str | None = None,
):
    """Writes the events to the c3d file.

    Args:
        c3d_path: The path to the original c3d file.
        event_table: The DataFrame containing the events.
        output_path: The path to write the c3d file with the events.
        If None, the original file will be overwritten.
    """
    io.C3dEventFileWriter(c3d_path).write_events(event_table, output_path)


def segment_trial(trial: model.Trial, method: str = "HS") -> model.TrialCycles:
    """Segments the trial into cycles

    Args:
        trial: The trial to segment.
        method: The method to use for segmenting the trial.
        Currently, only supports "HS" which segments the trial based on heel strikes.
        Default is "HS".

    Returns:
        The trial with the segmented data.
    """
    match method:
        case "HS":
            method_obj = segmentation.GaitEventsSegmentation()
        case _:
            raise ValueError(f"Unsupported method: {method}")

    trial_cycles = method_obj.segment(trial)
    return trial_cycles


def time_normalise_trial(
    trial: model.Trial | model.TrialCycles, method: str = "linear", **kwargs
) -> model.Trial | model.TrialCycles:
    """Normalises the time in the trial.

    Args:
        trial: The trial to normalise the time for.
        method: The method to use for normalising the time. Currently, only supports
        "linear" which normalises the time linearly. Default is "linear".

    Returns:
        The trial with the normalised time.
    """
    match method:
        case "linear":
            normaliser = normalisation.LinearTimeNormaliser(**kwargs)
        case _:
            raise ValueError(f"Unsupported method: {method}")

    trial = normaliser.normalise(trial)
    return trial
