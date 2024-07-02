from gaitalytics.input import (PointInputFileReader,
                               AnalogInputFileReader,
                               EventInputFileReader)
import gaitalytics.model as model


def map_point_data_category_input(
        input_reader: PointInputFileReader) -> model.DataCategory:
    """
    Maps the input data from a PointInputFileReader to a DataCategory object.

    Args:
        input_reader (PointInputFileReader): The input reader
         object that provides the point data.

    Returns:
        DataCategory: The mapped DataCategory object.

    """
    category = model.DataCategory(model.DataCategoryType.POINT,
                                  input_reader.get_points_frame_rate(),
                                  input_reader.get_points_times())
    for i in range(input_reader.get_points_size()):
        category.add_data_entity(input_reader.get_points_labels()[i],
                                 input_reader.get_point_data(i),
                                 input_reader.get_point_units()[i])
    return category


def map_analog_data_category_input(
        input_reader: AnalogInputFileReader) -> model.DataCategory:
    """
    Maps the input data from an AnalogInputFileReader to a DataCategory object.

    Args:
        input_reader (AnalogInputFileReader): The input reader
         object that provides the analog data.

    Returns:
        DataCategory: The mapped DataCategory object.

    """
    category = model.DataCategory(model.DataCategoryType.ANALOG,
                                  input_reader.get_analogs_frame_rate(),
                                  input_reader.get_analogs_times())
    for i in range(input_reader.get_analog_size()):
        category.add_data_entity(input_reader.get_analog_labels()[i],
                                 input_reader.get_analog_data(i),
                                 input_reader.get_analog_units()[i])
    return category


def map_events(input_reader: EventInputFileReader) -> model.Events:
    """
    Maps the input data from an EventInputFileReader to an Events object.

    Args:
        input_reader (EventInputFileReader): The input reader object
         that provides the event data.

    Returns:
        Events: The mapped Events object.

    """
    labels = input_reader.get_event_labels()
    times = input_reader.get_event_times()
    contexts = input_reader.get_event_contexts()
    events = model.Events(labels, times, contexts)
    return events


def map_trial(
        point_input_reader: PointInputFileReader,
        analog_input_reader: AnalogInputFileReader,
        event_input_reader) -> model.Trial:
    """
    Maps the input data from point, analog, and event input readers
    to create a Trial object.

    Args:
        point_input_reader (PointInputFileReader): The reader for point
         input data.
        analog_input_reader (AnalogInputFileReader): The reader for
         analog input data.
        event_input_reader: The reader for event input data.

    Returns:
        model.Trial: The mapped Trial object.

    """
    point_cat = map_point_data_category_input(point_input_reader)
    analog_cat = map_analog_data_category_input(analog_input_reader)
    events = map_events(event_input_reader)
    trial = model.Trial()
    trial.events = events
    trial.add_data_category(point_cat)
    trial.add_data_category(analog_cat)
    return trial
