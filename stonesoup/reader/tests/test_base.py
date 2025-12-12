"""Tests for base reader classes."""
import datetime
from pathlib import Path
from urllib.parse import ParseResult, urlparse

import pytest
import numpy as np

from ..base import (
    Reader, DetectionReader, GroundTruthReader, SensorDataReader,
    FrameReader, TrackReader
)
from ..file import FileReader, BinaryFileReader, TextFileReader
from ..url import UrlReader
from ...base import Property
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.track import Track
from ...types.state import State
from ...types.sensordata import SensorData, ImageFrame
from ...types.array import StateVector


# Concrete implementations for testing abstract base classes
class ConcreteDetectionReader(DetectionReader):
    """Concrete implementation of DetectionReader for testing."""
    detections_data: list = Property(default_factory=list, doc="Test detection data")

    @DetectionReader.generator_method
    def detections_gen(self):
        for timestamp, detections in self.detections_data:
            yield timestamp, detections


class ConcreteGroundTruthReader(GroundTruthReader):
    """Concrete implementation of GroundTruthReader for testing."""
    groundtruth_data: list = Property(default_factory=list, doc="Test ground truth data")

    @GroundTruthReader.generator_method
    def groundtruth_paths_gen(self):
        for timestamp, paths in self.groundtruth_data:
            yield timestamp, paths


class ConcreteSensorDataReader(SensorDataReader):
    """Concrete implementation of SensorDataReader for testing."""
    sensor_data_list: list = Property(default_factory=list, doc="Test sensor data")

    @SensorDataReader.generator_method
    def sensor_data_gen(self):
        for timestamp, data in self.sensor_data_list:
            yield timestamp, data


class ConcreteFrameReader(FrameReader):
    """Concrete implementation of FrameReader for testing."""
    frames_data: list = Property(default_factory=list, doc="Test frame data")

    @FrameReader.generator_method
    def frames_gen(self):
        for timestamp, frames in self.frames_data:
            yield timestamp, frames


class ConcreteTrackReader(TrackReader):
    """Concrete implementation of TrackReader for testing."""
    tracks_data: list = Property(default_factory=list, doc="Test track data")

    @TrackReader.generator_method
    def tracks_gen(self):
        for timestamp, tracks in self.tracks_data:
            yield timestamp, tracks


class ConcreteFileReader(FileReader):
    """Concrete implementation of FileReader for testing."""
    test_data: list = Property(default_factory=list, doc="Test data")

    @FileReader.generator_method
    def data_gen(self):
        for timestamp, data in self.test_data:
            yield timestamp, data


class ConcreteBinaryFileReader(BinaryFileReader):
    """Concrete implementation of BinaryFileReader for testing."""
    test_data: list = Property(default_factory=list, doc="Test data")

    @BinaryFileReader.generator_method
    def data_gen(self):
        for timestamp, data in self.test_data:
            yield timestamp, data


class ConcreteTextFileReader(TextFileReader):
    """Concrete implementation of TextFileReader for testing."""
    test_data: list = Property(default_factory=list, doc="Test data")

    @TextFileReader.generator_method
    def data_gen(self):
        for timestamp, data in self.test_data:
            yield timestamp, data


class ConcreteUrlReader(UrlReader):
    """Concrete implementation of UrlReader for testing."""
    test_data: list = Property(default_factory=list, doc="Test data")

    @UrlReader.generator_method
    def data_gen(self):
        for timestamp, data in self.test_data:
            yield timestamp, data


# Fixtures
@pytest.fixture
def sample_detections():
    """Create sample detection data."""
    timestamp1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    timestamp2 = datetime.datetime(2025, 1, 1, 12, 1, 0)

    det1 = Detection(StateVector([1.0, 2.0]), timestamp=timestamp1)
    det2 = Detection(StateVector([3.0, 4.0]), timestamp=timestamp2)
    det3 = Detection(StateVector([5.0, 6.0]), timestamp=timestamp2)

    return [
        (timestamp1, {det1}),
        (timestamp2, {det2, det3})
    ]


@pytest.fixture
def sample_groundtruth():
    """Create sample ground truth data."""
    timestamp1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    timestamp2 = datetime.datetime(2025, 1, 1, 12, 1, 0)

    gt_state1 = GroundTruthState(StateVector([1.0, 2.0]), timestamp=timestamp1)
    gt_state2 = GroundTruthState(StateVector([3.0, 4.0]), timestamp=timestamp2)

    gt_path1 = GroundTruthPath([gt_state1])
    gt_path2 = GroundTruthPath([gt_state2])

    return [
        (timestamp1, {gt_path1}),
        (timestamp2, {gt_path2})
    ]


@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data."""
    timestamp1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    timestamp2 = datetime.datetime(2025, 1, 1, 12, 1, 0)

    # SensorData is just a base type, use ImageFrame which is a concrete implementation
    sensor_data1 = ImageFrame(pixels=np.zeros((10, 10, 3)), timestamp=timestamp1)
    sensor_data2 = ImageFrame(pixels=np.ones((10, 10, 3)), timestamp=timestamp2)

    return [
        (timestamp1, {sensor_data1}),
        (timestamp2, {sensor_data2})
    ]


@pytest.fixture
def sample_frames():
    """Create sample frame data."""
    timestamp1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    timestamp2 = datetime.datetime(2025, 1, 1, 12, 1, 0)

    frame1 = ImageFrame(np.zeros((10, 10, 3)), timestamp=timestamp1)
    frame2 = ImageFrame(np.ones((10, 10, 3)), timestamp=timestamp2)

    return [
        (timestamp1, {frame1}),
        (timestamp2, {frame2})
    ]


@pytest.fixture
def sample_tracks():
    """Create sample track data."""
    timestamp1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    timestamp2 = datetime.datetime(2025, 1, 1, 12, 1, 0)

    state1 = State(StateVector([1.0, 2.0]), timestamp=timestamp1)
    state2 = State(StateVector([3.0, 4.0]), timestamp=timestamp2)

    track1 = Track([state1])
    track2 = Track([state2])

    return [
        (timestamp1, {track1}),
        (timestamp2, {track2})
    ]


# Tests for Reader base class
def test_reader_is_abstract():
    """Test that Reader is abstract and cannot be instantiated directly."""
    # Reader itself doesn't have abstract methods, but it's a base class
    # It can be instantiated but won't be useful without a generator method
    reader = Reader()
    assert isinstance(reader, Reader)
    # Should raise AttributeError when trying to iterate without generator method
    with pytest.raises(AttributeError, match="Generator method undefined"):
        list(reader)


# Tests for DetectionReader
def test_detection_reader_instantiation(sample_detections):
    """Test DetectionReader can be instantiated with concrete implementation."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)
    assert isinstance(reader, DetectionReader)
    assert isinstance(reader, Reader)


def test_detection_reader_iteration(sample_detections):
    """Test DetectionReader iterator protocol."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    results = list(reader)
    assert len(results) == 2

    timestamp1, detections1 = results[0]
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(detections1) == 1

    timestamp2, detections2 = results[1]
    assert timestamp2 == datetime.datetime(2025, 1, 1, 12, 1, 0)
    assert len(detections2) == 2


def test_detection_reader_detections_property(sample_detections):
    """Test DetectionReader.detections property access."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    # Iterate to first element
    next(iter(reader))
    detections = reader.detections
    assert len(detections) == 1
    assert all(isinstance(det, Detection) for det in detections)


def test_detection_reader_current_property(sample_detections):
    """Test DetectionReader.current property access."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    # Iterate to first element
    next(iter(reader))
    timestamp, detections = reader.current
    assert timestamp == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(detections) == 1


def test_detection_reader_empty():
    """Test DetectionReader with no data."""
    reader = ConcreteDetectionReader([])
    results = list(reader)
    assert len(results) == 0


def test_detection_reader_abstract_method():
    """Test that detections_gen is abstract and must be implemented."""
    # Create a class without implementing detections_gen
    class IncompleteDetectionReader(DetectionReader):
        pass

    # Should not be able to use it
    with pytest.raises(TypeError):
        IncompleteDetectionReader()


# Tests for GroundTruthReader
def test_groundtruth_reader_instantiation(sample_groundtruth):
    """Test GroundTruthReader can be instantiated with concrete implementation."""
    reader = ConcreteGroundTruthReader(groundtruth_data=sample_groundtruth)
    assert isinstance(reader, GroundTruthReader)
    assert isinstance(reader, Reader)


def test_groundtruth_reader_iteration(sample_groundtruth):
    """Test GroundTruthReader iterator protocol."""
    reader = ConcreteGroundTruthReader(groundtruth_data=sample_groundtruth)

    results = list(reader)
    assert len(results) == 2

    timestamp1, paths1 = results[0]
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(paths1) == 1

    timestamp2, paths2 = results[1]
    assert timestamp2 == datetime.datetime(2025, 1, 1, 12, 1, 0)
    assert len(paths2) == 1


def test_groundtruth_reader_paths_property(sample_groundtruth):
    """Test GroundTruthReader.groundtruth_paths property access."""
    reader = ConcreteGroundTruthReader(groundtruth_data=sample_groundtruth)

    # Iterate to first element
    next(iter(reader))
    paths = reader.groundtruth_paths
    assert len(paths) == 1
    assert all(isinstance(path, GroundTruthPath) for path in paths)


def test_groundtruth_reader_empty():
    """Test GroundTruthReader with no data."""
    reader = ConcreteGroundTruthReader([])
    results = list(reader)
    assert len(results) == 0


def test_groundtruth_reader_abstract_method():
    """Test that groundtruth_paths_gen is abstract and must be implemented."""
    class IncompleteGroundTruthReader(GroundTruthReader):
        pass

    with pytest.raises(TypeError):
        IncompleteGroundTruthReader()


# Tests for SensorDataReader
def test_sensor_data_reader_instantiation(sample_sensor_data):
    """Test SensorDataReader can be instantiated with concrete implementation."""
    reader = ConcreteSensorDataReader(sensor_data_list=sample_sensor_data)
    assert isinstance(reader, SensorDataReader)
    assert isinstance(reader, Reader)


def test_sensor_data_reader_iteration(sample_sensor_data):
    """Test SensorDataReader iterator protocol."""
    reader = ConcreteSensorDataReader(sensor_data_list=sample_sensor_data)

    results = list(reader)
    assert len(results) == 2

    timestamp1, data1 = results[0]
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(data1) == 1

    timestamp2, data2 = results[1]
    assert timestamp2 == datetime.datetime(2025, 1, 1, 12, 1, 0)
    assert len(data2) == 1


def test_sensor_data_reader_property(sample_sensor_data):
    """Test SensorDataReader.sensor_data property access."""
    reader = ConcreteSensorDataReader(sensor_data_list=sample_sensor_data)

    # Iterate to first element
    next(iter(reader))
    sensor_data = reader.sensor_data
    assert len(sensor_data) == 1
    # ImageFrame is a subclass of SensorData
    assert all(isinstance(data, (SensorData, ImageFrame)) for data in sensor_data)


def test_sensor_data_reader_empty():
    """Test SensorDataReader with no data."""
    reader = ConcreteSensorDataReader([])
    results = list(reader)
    assert len(results) == 0


def test_sensor_data_reader_abstract_method():
    """Test that sensor_data_gen is abstract and must be implemented."""
    class IncompleteSensorDataReader(SensorDataReader):
        pass

    with pytest.raises(TypeError):
        IncompleteSensorDataReader()


# Tests for FrameReader
def test_frame_reader_instantiation(sample_frames):
    """Test FrameReader can be instantiated with concrete implementation."""
    reader = ConcreteFrameReader(frames_data=sample_frames)
    assert isinstance(reader, FrameReader)
    assert isinstance(reader, SensorDataReader)
    assert isinstance(reader, Reader)


def test_frame_reader_iteration(sample_frames):
    """Test FrameReader iterator protocol."""
    reader = ConcreteFrameReader(frames_data=sample_frames)

    results = list(reader)
    assert len(results) == 2

    timestamp1, frames1 = results[0]
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(frames1) == 1

    timestamp2, frames2 = results[1]
    assert timestamp2 == datetime.datetime(2025, 1, 1, 12, 1, 0)
    assert len(frames2) == 1


def test_frame_reader_frame_property(sample_frames):
    """Test FrameReader.frame property access."""
    reader = ConcreteFrameReader(frames_data=sample_frames)

    # Iterate to first element
    next(iter(reader))
    frames = reader.frame
    assert len(frames) == 1
    assert all(isinstance(frame, ImageFrame) for frame in frames)


def test_frame_reader_sensor_data_property(sample_frames):
    """Test FrameReader.sensor_data property access (inherited)."""
    reader = ConcreteFrameReader(frames_data=sample_frames)

    # Iterate to first element
    next(iter(reader))
    sensor_data = reader.sensor_data
    assert len(sensor_data) == 1
    assert all(isinstance(data, ImageFrame) for data in sensor_data)


def test_frame_reader_sensor_data_gen_wrapper(sample_frames):
    """Test FrameReader.sensor_data_gen wraps frames_gen."""
    reader = ConcreteFrameReader(frames_data=sample_frames)

    # Manually call sensor_data_gen
    gen = reader.sensor_data_gen()
    timestamp1, frames1 = next(gen)
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(frames1) == 1
    assert all(isinstance(frame, ImageFrame) for frame in frames1)


def test_frame_reader_empty():
    """Test FrameReader with no data."""
    reader = ConcreteFrameReader([])
    results = list(reader)
    assert len(results) == 0


def test_frame_reader_abstract_method():
    """Test that frames_gen is abstract and must be implemented."""
    class IncompleteFrameReader(FrameReader):
        pass

    with pytest.raises(TypeError):
        IncompleteFrameReader()


# Tests for TrackReader
def test_track_reader_instantiation(sample_tracks):
    """Test TrackReader can be instantiated with concrete implementation."""
    reader = ConcreteTrackReader(tracks_data=sample_tracks)
    assert isinstance(reader, TrackReader)
    assert isinstance(reader, Reader)


def test_track_reader_iteration(sample_tracks):
    """Test TrackReader iterator protocol."""
    reader = ConcreteTrackReader(tracks_data=sample_tracks)

    results = list(reader)
    assert len(results) == 2

    timestamp1, tracks1 = results[0]
    assert timestamp1 == datetime.datetime(2025, 1, 1, 12, 0, 0)
    assert len(tracks1) == 1

    timestamp2, tracks2 = results[1]
    assert timestamp2 == datetime.datetime(2025, 1, 1, 12, 1, 0)
    assert len(tracks2) == 1


def test_track_reader_tracks_property(sample_tracks):
    """Test TrackReader.tracks property access."""
    reader = ConcreteTrackReader(tracks_data=sample_tracks)

    # Iterate to first element
    next(iter(reader))
    tracks = reader.tracks
    assert len(tracks) == 1
    assert all(isinstance(track, Track) for track in tracks)


def test_track_reader_empty():
    """Test TrackReader with no data."""
    reader = ConcreteTrackReader([])
    results = list(reader)
    assert len(results) == 0


def test_track_reader_abstract_method():
    """Test that tracks_gen is abstract and must be implemented."""
    class IncompleteTrackReader(TrackReader):
        pass

    with pytest.raises(TypeError):
        IncompleteTrackReader()


# Tests for FileReader
def test_file_reader_path_conversion(tmpdir):
    """Test FileReader converts string path to Path object."""
    test_file = tmpdir.join("test.txt")
    test_file.write("test content")

    # Pass as string
    reader = ConcreteFileReader(str(test_file))
    assert isinstance(reader.path, Path)
    assert reader.path == Path(str(test_file))


def test_file_reader_path_object(tmpdir):
    """Test FileReader accepts Path object."""
    test_file = tmpdir.join("test.txt")
    test_file.write("test content")

    # Pass as Path
    path_obj = Path(str(test_file))
    reader = ConcreteFileReader(path_obj)
    assert isinstance(reader.path, Path)
    assert reader.path == path_obj


def test_file_reader_iteration(tmpdir):
    """Test FileReader iteration with test data."""
    test_file = tmpdir.join("test.txt")
    test_file.write("test content")

    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0)
    test_data = [(timestamp, "data")]

    reader = ConcreteFileReader(path=str(test_file), test_data=test_data)

    results = list(reader)
    assert len(results) == 1
    assert results[0] == (timestamp, "data")


def test_file_reader_nonexistent_path():
    """Test FileReader with non-existent path (should still create Path object)."""
    reader = ConcreteFileReader("/nonexistent/path/file.txt")
    assert isinstance(reader.path, Path)
    assert reader.path == Path("/nonexistent/path/file.txt")


# Tests for BinaryFileReader
def test_binary_file_reader_instantiation(tmpdir):
    """Test BinaryFileReader can be instantiated."""
    test_file = tmpdir.join("test.bin")
    test_file.write_binary(b"binary content")

    reader = ConcreteBinaryFileReader(str(test_file))
    assert isinstance(reader, BinaryFileReader)
    assert isinstance(reader, FileReader)
    assert isinstance(reader.path, Path)


def test_binary_file_reader_path_conversion(tmpdir):
    """Test BinaryFileReader converts string path to Path object."""
    test_file = tmpdir.join("test.bin")
    test_file.write_binary(b"binary content")

    reader = ConcreteBinaryFileReader(str(test_file))
    assert isinstance(reader.path, Path)
    assert reader.path == Path(str(test_file))


def test_binary_file_reader_iteration(tmpdir):
    """Test BinaryFileReader iteration with test data."""
    test_file = tmpdir.join("test.bin")
    test_file.write_binary(b"binary content")

    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0)
    test_data = [(timestamp, b"data")]

    reader = ConcreteBinaryFileReader(path=str(test_file), test_data=test_data)

    results = list(reader)
    assert len(results) == 1
    assert results[0] == (timestamp, b"data")


# Tests for TextFileReader
def test_text_file_reader_instantiation(tmpdir):
    """Test TextFileReader can be instantiated."""
    test_file = tmpdir.join("test.txt")
    test_file.write("text content")

    reader = ConcreteTextFileReader(str(test_file))
    assert isinstance(reader, TextFileReader)
    assert isinstance(reader, FileReader)
    assert isinstance(reader.path, Path)


def test_text_file_reader_default_encoding(tmpdir):
    """Test TextFileReader has default UTF-8 encoding."""
    test_file = tmpdir.join("test.txt")
    test_file.write("text content")

    reader = ConcreteTextFileReader(str(test_file))
    assert reader.encoding == "utf-8"


def test_text_file_reader_custom_encoding(tmpdir):
    """Test TextFileReader can specify custom encoding."""
    test_file = tmpdir.join("test.txt")
    test_file.write("text content")

    reader = ConcreteTextFileReader(str(test_file), encoding="ascii")
    assert reader.encoding == "ascii"


def test_text_file_reader_path_conversion(tmpdir):
    """Test TextFileReader converts string path to Path object."""
    test_file = tmpdir.join("test.txt")
    test_file.write("text content")

    reader = ConcreteTextFileReader(str(test_file))
    assert isinstance(reader.path, Path)
    assert reader.path == Path(str(test_file))


def test_text_file_reader_iteration(tmpdir):
    """Test TextFileReader iteration with test data."""
    test_file = tmpdir.join("test.txt")
    test_file.write("text content")

    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0)
    test_data = [(timestamp, "data")]

    reader = ConcreteTextFileReader(path=str(test_file), test_data=test_data)

    results = list(reader)
    assert len(results) == 1
    assert results[0] == (timestamp, "data")


# Tests for UrlReader
def test_url_reader_url_conversion():
    """Test UrlReader converts string URL to ParseResult object."""
    url_string = "http://example.com/data.json"

    reader = ConcreteUrlReader(url_string)
    assert isinstance(reader.url, ParseResult)
    assert reader.url.scheme == "http"
    assert reader.url.netloc == "example.com"
    assert reader.url.path == "/data.json"


def test_url_reader_parse_result():
    """Test UrlReader accepts ParseResult object."""
    url_string = "https://example.com:8080/path/to/resource?query=value#fragment"
    parsed_url = urlparse(url_string)

    reader = ConcreteUrlReader(parsed_url)
    assert isinstance(reader.url, ParseResult)
    assert reader.url == parsed_url
    assert reader.url.scheme == "https"
    assert reader.url.netloc == "example.com:8080"
    assert reader.url.path == "/path/to/resource"
    assert reader.url.query == "query=value"
    assert reader.url.fragment == "fragment"


def test_url_reader_iteration():
    """Test UrlReader iteration with test data."""
    url_string = "http://example.com/data.json"
    timestamp = datetime.datetime(2025, 1, 1, 12, 0, 0)
    test_data = [(timestamp, {"key": "value"})]

    reader = ConcreteUrlReader(url=url_string, test_data=test_data)

    results = list(reader)
    assert len(results) == 1
    assert results[0] == (timestamp, {"key": "value"})


def test_url_reader_various_schemes():
    """Test UrlReader handles various URL schemes."""
    schemes = [
        "http://example.com",
        "https://example.com",
        "ftp://ftp.example.com",
        "file:///path/to/file",
    ]

    for url_string in schemes:
        reader = ConcreteUrlReader(url_string)
        assert isinstance(reader.url, ParseResult)
        parsed = urlparse(url_string)
        assert reader.url.scheme == parsed.scheme


def test_url_reader_url_components():
    """Test UrlReader preserves all URL components."""
    url_string = "https://user:pass@example.com:8080/path?query=1#fragment"
    reader = ConcreteUrlReader(url_string)

    assert reader.url.scheme == "https"
    assert reader.url.netloc == "user:pass@example.com:8080"
    assert reader.url.path == "/path"
    assert reader.url.query == "query=1"
    assert reader.url.fragment == "fragment"


# Integration tests
def test_multiple_readers_iteration(sample_detections, sample_tracks):
    """Test that multiple readers can be used independently."""
    det_reader = ConcreteDetectionReader(detections_data=sample_detections)
    track_reader = ConcreteTrackReader(tracks_data=sample_tracks)

    det_results = list(det_reader)
    track_results = list(track_reader)

    assert len(det_results) == 2
    assert len(track_results) == 2


def test_reader_reusability(sample_detections):
    """Test that reader can be iterated multiple times."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    # First iteration
    results1 = list(reader)
    assert len(results1) == 2

    # Second iteration - should work again
    results2 = list(reader)
    assert len(results2) == 2
    assert results1 == results2


def test_reader_current_state_updates(sample_detections):
    """Test that current state updates as iteration proceeds."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    iterator = iter(reader)

    # First iteration
    first_result = next(iterator)
    first_current = reader.current
    assert first_result == first_current

    # Second iteration
    second_result = next(iterator)
    second_current = reader.current
    assert second_result == second_current
    assert first_current != second_current


def test_reader_partial_iteration(sample_detections):
    """Test that reader can be partially iterated."""
    reader = ConcreteDetectionReader(detections_data=sample_detections)

    iterator = iter(reader)

    # Only get first element
    first_result = next(iterator)
    assert first_result == sample_detections[0]

    # Current should be set to first element
    assert reader.current == first_result


def test_file_reader_with_relative_path(tmpdir):
    """Test FileReader with relative path."""
    # Change to tmpdir to use relative paths
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(str(tmpdir))
        test_file = tmpdir.join("test.txt")
        test_file.write("test content")

        reader = ConcreteFileReader("test.txt")
        assert isinstance(reader.path, Path)
        assert reader.path.name == "test.txt"
    finally:
        os.chdir(original_cwd)


def test_text_file_reader_with_multiple_encodings(tmpdir):
    """Test TextFileReader can handle different encodings."""
    test_file = tmpdir.join("test.txt")
    test_file.write("test content")

    encodings = ["utf-8", "ascii", "latin-1", "utf-16"]

    for encoding in encodings:
        reader = ConcreteTextFileReader(str(test_file), encoding=encoding)
        assert reader.encoding == encoding
