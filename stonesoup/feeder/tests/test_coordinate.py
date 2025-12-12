"""Tests for coordinate system conversion feeders."""

import datetime

import numpy as np
import pytest

from ..coordinate import (
    GeodeticToECEFConverter,
    ECEFToGeodeticConverter,
    ECIToECEFConverter,
    ECEFToECIConverter,
    GeodeticToECIConverter,
    ECIToGeodeticConverter,
)
from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader, GroundTruthReader
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthState, GroundTruthPath
from ...types.coordinates import WGS84, GRS80
from ...functions.coordinates import geodetic_to_ecef, ecef_to_geodetic


@pytest.fixture()
def geodetic_detector():
    """Detector with geodetic coordinates (lat, lon, alt) in radians."""
    class Detector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            time_step = datetime.timedelta(seconds=1)

            # Greenwich Observatory (approximately)
            lat = np.radians(51.4769)
            lon = np.radians(-0.0005)
            alt = 0.0
            yield time, {Detection([[lat], [lon], [alt]], timestamp=time)}

            # Equator on prime meridian
            time += time_step
            lat = 0.0
            lon = 0.0
            alt = 1000.0
            yield time, {Detection([[lat], [lon], [alt]], timestamp=time)}

            # North pole
            time += time_step
            lat = np.pi / 2
            lon = 0.0
            alt = 0.0
            yield time, {Detection([[lat], [lon], [alt]], timestamp=time)}

            # Sydney, Australia
            time += time_step
            lat = np.radians(-33.8688)
            lon = np.radians(151.2093)
            alt = 58.0
            yield time, {Detection([[lat], [lon], [alt]], timestamp=time)}

    return Detector()


@pytest.fixture()
def ecef_detector():
    """Detector with ECEF coordinates (x, y, z) in meters."""
    class Detector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            time_step = datetime.timedelta(seconds=1)

            # Point on equator at prime meridian
            yield time, {Detection([[WGS84.semi_major_axis], [0.0], [0.0]], timestamp=time)}

            # Point on equator at 90°E
            time += time_step
            yield time, {Detection([[0.0], [WGS84.semi_major_axis], [0.0]], timestamp=time)}

            # North pole
            time += time_step
            yield time, {Detection([[0.0], [0.0], [WGS84.semi_minor_axis]], timestamp=time)}

            # Generic point
            time += time_step
            yield time, {Detection([[4000000.0], [3000000.0], [4500000.0]], timestamp=time)}

    return Detector()


@pytest.fixture()
def eci_detector():
    """Detector with ECI coordinates (x, y, z) in meters."""
    class Detector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            time_step = datetime.timedelta(seconds=1)

            # Point at Earth radius
            yield time, {Detection([[7000000.0], [0.0], [0.0]], timestamp=time)}

            time += time_step
            yield time, {Detection([[0.0], [7000000.0], [0.0]], timestamp=time)}

            time += time_step
            yield time, {Detection([[5000000.0], [5000000.0], [1000000.0]], timestamp=time)}

    return Detector()


@pytest.fixture()
def geodetic_groundtruth():
    """Ground truth reader with geodetic coordinates."""
    class GroundTruth(GroundTruthReader):
        @BufferedGenerator.generator_method
        def groundtruth_paths_gen(self):
            time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            time_step = datetime.timedelta(seconds=1)
            path = GroundTruthPath()

            # Greenwich Observatory
            lat = np.radians(51.4769)
            lon = np.radians(-0.0005)
            alt = 0.0
            state = GroundTruthState([[lat], [lon], [alt]], timestamp=time)
            path.append(state)
            yield time, {path}

            # Move slightly
            time += time_step
            lat = np.radians(51.4770)
            lon = np.radians(-0.0004)
            alt = 10.0
            state = GroundTruthState([[lat], [lon], [alt]], timestamp=time)
            path.append(state)
            yield time, {path}

    return GroundTruth()


# GeodeticToECEFConverter tests

def test_geodetic_to_ecef_instantiation(geodetic_detector):
    """Test that GeodeticToECEFConverter can be instantiated."""
    converter = GeodeticToECEFConverter(reader=geodetic_detector)
    assert converter.mapping == (0, 1, 2)
    assert converter.ellipsoid is WGS84


def test_geodetic_to_ecef_custom_params(geodetic_detector):
    """Test GeodeticToECEFConverter with custom parameters."""
    converter = GeodeticToECEFConverter(
        reader=geodetic_detector,
        mapping=(2, 0, 1),
        ellipsoid=GRS80
    )
    assert converter.mapping == (2, 0, 1)
    assert converter.ellipsoid is GRS80


def test_geodetic_to_ecef_equator_prime_meridian(geodetic_detector):
    """Test geodetic to ECEF conversion at equator on prime meridian."""
    converter = GeodeticToECEFConverter(reader=geodetic_detector)

    # Skip first detection, get second (equator at prime meridian)
    gen = iter(converter)
    next(gen)
    time, detections = next(gen)

    detection = detections.pop()
    x = detection.state_vector[0, 0]
    y = detection.state_vector[1, 0]
    z = detection.state_vector[2, 0]

    # At equator, prime meridian: x should be about semi-major axis + altitude
    expected_x = WGS84.semi_major_axis + 1000.0
    assert pytest.approx(x, abs=1.0) == expected_x
    assert pytest.approx(y, abs=1.0) == 0.0
    assert pytest.approx(z, abs=1.0) == 0.0


def test_geodetic_to_ecef_north_pole(geodetic_detector):
    """Test geodetic to ECEF conversion at north pole."""
    converter = GeodeticToECEFConverter(reader=geodetic_detector)

    # Skip to third detection (north pole)
    gen = iter(converter)
    next(gen)
    next(gen)
    time, detections = next(gen)

    detection = detections.pop()
    x = detection.state_vector[0, 0]
    y = detection.state_vector[1, 0]
    z = detection.state_vector[2, 0]

    # At north pole: x and y should be ~0, z should be ~semi_minor_axis
    assert pytest.approx(x, abs=1.0) == 0.0
    assert pytest.approx(y, abs=1.0) == 0.0
    assert pytest.approx(z, abs=1.0) == WGS84.semi_minor_axis


def test_geodetic_to_ecef_round_trip():
    """Test that geodetic → ECEF → geodetic round trip preserves coordinates."""
    # Test with known geodetic coordinates
    test_cases = [
        (0.0, 0.0, 0.0),  # Equator, prime meridian
        (np.radians(45.0), np.radians(90.0), 1000.0),  # Mid-latitude
        (np.radians(-33.8688), np.radians(151.2093), 58.0),  # Sydney
    ]

    for lat, lon, alt in test_cases:
        # Convert to ECEF
        ecef = geodetic_to_ecef(lat, lon, alt, WGS84)

        # Convert back to geodetic
        lat2, lon2, alt2 = ecef_to_geodetic(ecef[0], ecef[1], ecef[2], WGS84)

        # Should match original within numerical precision
        # 1e-6 radians ≈ 6 meters at Earth's surface (Bowring's iterative method)
        assert pytest.approx(lat2, abs=1e-6) == lat
        assert pytest.approx(lon2, abs=1e-6) == lon
        assert pytest.approx(alt2, abs=1.0) == alt  # Altitude within 1 meter


# ECEFToGeodeticConverter tests

def test_ecef_to_geodetic_instantiation(ecef_detector):
    """Test that ECEFToGeodeticConverter can be instantiated."""
    converter = ECEFToGeodeticConverter(reader=ecef_detector)
    assert converter.mapping == (0, 1, 2)
    assert converter.ellipsoid is WGS84


def test_ecef_to_geodetic_equator_prime_meridian(ecef_detector):
    """Test ECEF to geodetic conversion at equator on prime meridian."""
    converter = ECEFToGeodeticConverter(reader=ecef_detector)

    time, detections = next(iter(converter))
    detection = detections.pop()

    lat = detection.state_vector[0, 0]
    lon = detection.state_vector[1, 0]
    alt = detection.state_vector[2, 0]

    # At equator, prime meridian
    assert pytest.approx(lat, abs=1e-10) == 0.0
    assert pytest.approx(lon, abs=1e-10) == 0.0
    assert pytest.approx(alt, abs=1.0) == 0.0


def test_ecef_to_geodetic_equator_90_east(ecef_detector):
    """Test ECEF to geodetic conversion at equator, 90° East."""
    converter = ECEFToGeodeticConverter(reader=ecef_detector)

    gen = iter(converter)
    next(gen)
    time, detections = next(gen)
    detection = detections.pop()

    lat = detection.state_vector[0, 0]
    lon = detection.state_vector[1, 0]
    alt = detection.state_vector[2, 0]

    # At equator, 90°E
    assert pytest.approx(lat, abs=1e-10) == 0.0
    assert pytest.approx(lon, abs=1e-6) == np.pi / 2
    assert pytest.approx(alt, abs=1.0) == 0.0


def test_ecef_to_geodetic_north_pole(ecef_detector):
    """Test ECEF to geodetic conversion at north pole."""
    converter = ECEFToGeodeticConverter(reader=ecef_detector)

    gen = iter(converter)
    next(gen)
    next(gen)
    time, detections = next(gen)
    detection = detections.pop()

    lat = detection.state_vector[0, 0]
    alt = detection.state_vector[2, 0]

    # At north pole
    assert pytest.approx(lat, abs=1e-6) == np.pi / 2
    assert pytest.approx(alt, abs=1.0) == 0.0


# ECIToECEFConverter tests

def test_eci_to_ecef_instantiation(eci_detector):
    """Test that ECIToECEFConverter can be instantiated."""
    converter = ECIToECEFConverter(reader=eci_detector)
    assert converter.mapping == (0, 1, 2)


def test_eci_to_ecef_preserves_magnitude(eci_detector):
    """Test that ECI→ECEF preserves position magnitude."""
    converter = ECIToECEFConverter(reader=eci_detector)

    for time, detections in converter:
        for detection in detections:
            x = detection.state_vector[0, 0]
            y = detection.state_vector[1, 0]
            z = detection.state_vector[2, 0]

            # Position magnitude should be preserved (rotation only)
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            # Check it's in reasonable range (7 million meters)
            assert magnitude > 1e6
            assert magnitude < 1e8


def test_eci_to_ecef_z_unchanged(eci_detector):
    """Test that Z coordinate is unchanged (rotation about Z-axis)."""
    converter = ECIToECEFConverter(reader=eci_detector)

    gen = iter(converter)
    time, detections = next(gen)
    detection = detections.pop()

    # Z should be unchanged (0.0 for first detection)
    z = detection.state_vector[2, 0]
    assert pytest.approx(z, abs=1e-6) == 0.0


def test_eci_to_ecef_requires_timestamp():
    """Test that ECI to ECEF converter requires timestamp on states."""
    class NoTimestampDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            yield None, {Detection([[1000000.0], [0.0], [0.0]])}

    converter = ECIToECEFConverter(reader=NoTimestampDetector())

    with pytest.raises(ValueError, match="requires a timestamp"):
        next(iter(converter))


# ECEFToECIConverter tests

def test_ecef_to_eci_instantiation(ecef_detector):
    """Test that ECEFToECIConverter can be instantiated."""
    converter = ECEFToECIConverter(reader=ecef_detector)
    assert converter.mapping == (0, 1, 2)


def test_eci_ecef_round_trip():
    """Test that ECI → ECEF → ECI is identity using coordinate functions."""
    from ...functions.coordinates import eci_to_ecef, ecef_to_eci

    timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0)

    # Test cases
    test_coords = [
        np.array([7000000.0, 0.0, 0.0]),
        np.array([0.0, 7000000.0, 0.0]),
        np.array([5000000.0, 5000000.0, 1000000.0]),
    ]

    for eci_original in test_coords:
        # Convert ECI → ECEF
        ecef = eci_to_ecef(eci_original, timestamp)

        # Convert ECEF → ECI
        eci_recovered = ecef_to_eci(ecef, timestamp)

        # Should match original within numerical precision
        # Use abs tolerance for sub-meter accuracy on ~7000km positions
        assert pytest.approx(eci_recovered[0], abs=1e-6) == eci_original[0]
        assert pytest.approx(eci_recovered[1], abs=1e-6) == eci_original[1]
        assert pytest.approx(eci_recovered[2], abs=1e-6) == eci_original[2]


# GeodeticToECIConverter tests

def test_geodetic_to_eci_instantiation(geodetic_detector):
    """Test that GeodeticToECIConverter can be instantiated."""
    converter = GeodeticToECIConverter(reader=geodetic_detector)
    assert converter.mapping == (0, 1, 2)
    assert converter.ellipsoid is WGS84


def test_geodetic_to_eci_produces_valid_coordinates(geodetic_detector):
    """Test that geodetic to ECI conversion produces valid coordinates."""
    converter = GeodeticToECIConverter(reader=geodetic_detector)

    for time, detections in converter:
        for detection in detections:
            x = detection.state_vector[0, 0]
            y = detection.state_vector[1, 0]
            z = detection.state_vector[2, 0]

            # Position magnitude should be in reasonable range
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            assert magnitude > WGS84.semi_minor_axis * 0.9
            assert magnitude < WGS84.semi_major_axis * 1.5


def test_geodetic_to_eci_requires_timestamp():
    """Test that geodetic to ECI converter requires timestamp on states."""
    class NoTimestampDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            yield None, {Detection([[0.0], [0.0], [0.0]])}

    converter = GeodeticToECIConverter(reader=NoTimestampDetector())

    with pytest.raises(ValueError, match="requires a timestamp"):
        next(iter(converter))


# ECIToGeodeticConverter tests

def test_eci_to_geodetic_instantiation(eci_detector):
    """Test that ECIToGeodeticConverter can be instantiated."""
    converter = ECIToGeodeticConverter(reader=eci_detector)
    assert converter.mapping == (0, 1, 2)
    assert converter.ellipsoid is WGS84


def test_eci_to_geodetic_produces_valid_coordinates(eci_detector):
    """Test that ECI to geodetic conversion produces valid coordinates."""
    converter = ECIToGeodeticConverter(reader=eci_detector)

    for time, detections in converter:
        for detection in detections:
            lat = detection.state_vector[0, 0]
            lon = detection.state_vector[1, 0]
            alt = detection.state_vector[2, 0]

            # Latitude should be in valid range
            assert -np.pi / 2 <= lat <= np.pi / 2

            # Longitude should be in valid range
            assert -np.pi <= lon <= np.pi

            # Altitude should be reasonable for near-Earth positions
            assert alt > -1e6
            assert alt < 1e7


# Ground truth tests

def test_geodetic_to_ecef_groundtruth(geodetic_groundtruth):
    """Test GeodeticToECEFConverter with ground truth."""
    converter = GeodeticToECEFConverter(reader=geodetic_groundtruth)

    paths_seen = 0
    for time, paths in converter:
        paths_seen += 1
        for path in paths:
            # Check that states in path have been converted
            for state in path:
                x = state.state_vector[0, 0]
                y = state.state_vector[1, 0]
                z = state.state_vector[2, 0]

                # Should be valid ECEF coordinates (near Earth)
                magnitude = np.sqrt(x**2 + y**2 + z**2)
                assert magnitude > WGS84.semi_minor_axis * 0.9
                assert magnitude < WGS84.semi_major_axis * 1.5

    assert paths_seen == 2


# Custom mapping tests

def test_geodetic_to_ecef_custom_mapping():
    """Test GeodeticToECEFConverter with non-standard mapping."""
    class CustomDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2024, 1, 1, 12, 0, 0)
            # State vector: [some_value, lat, some_other, lon, alt]
            lat = np.radians(45.0)
            lon = np.radians(90.0)
            alt = 1000.0
            yield time, {Detection([[999.0], [lat], [888.0], [lon], [alt]], timestamp=time)}

    converter = GeodeticToECEFConverter(
        reader=CustomDetector(),
        mapping=(1, 3, 4)  # lat at index 1, lon at 3, alt at 4
    )

    time, detections = next(iter(converter))
    detection = detections.pop()

    # Original values at indices 0 and 2 should be unchanged
    assert detection.state_vector[0, 0] == 999.0
    assert detection.state_vector[2, 0] == 888.0

    # Converted values at indices 1, 3, 4
    x = detection.state_vector[1, 0]
    y = detection.state_vector[3, 0]
    z = detection.state_vector[4, 0]

    # Verify conversion is correct
    expected = geodetic_to_ecef(np.radians(45.0), np.radians(90.0), 1000.0)
    assert pytest.approx(x, abs=1.0) == expected[0]
    assert pytest.approx(y, abs=1.0) == expected[1]
    assert pytest.approx(z, abs=1.0) == expected[2]
