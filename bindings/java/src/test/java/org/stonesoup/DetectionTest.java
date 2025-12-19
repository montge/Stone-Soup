package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Detection}.
 */
@DisplayName("Detection")
class DetectionTest {

    private static final double EPSILON = 1e-9;

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates detection from StateVector")
        void createsDetectionFromStateVector() {
            StateVector measurement = new StateVector(new double[]{100.0, 200.0});
            Detection detection = new Detection(measurement, 1.5);

            assertEquals(measurement, detection.getMeasurement());
            assertEquals(1.5, detection.getTimestamp(), EPSILON);
        }

        @Test
        @DisplayName("creates detection from array")
        void createsDetectionFromArray() {
            Detection detection = Detection.of(new double[]{50.0, 75.0, 25.0}, 2.0);

            assertEquals(3, detection.getDim());
            assertEquals(50.0, detection.get(0), EPSILON);
            assertEquals(75.0, detection.get(1), EPSILON);
            assertEquals(25.0, detection.get(2), EPSILON);
            assertEquals(2.0, detection.getTimestamp(), EPSILON);
        }

        @Test
        @DisplayName("rejects null measurement")
        void rejectsNullMeasurement() {
            assertThrows(NullPointerException.class,
                () -> new Detection(null, 0.0));
        }
    }

    @Nested
    @DisplayName("Accessors")
    class Accessors {

        @Test
        @DisplayName("gets measurement dimension")
        void getsMeasurementDimension() {
            Detection detection = Detection.of(new double[]{1.0, 2.0, 3.0, 4.0}, 0.0);
            assertEquals(4, detection.getDim());
        }

        @Test
        @DisplayName("gets measurement by index")
        void getsMeasurementByIndex() {
            Detection detection = Detection.of(new double[]{10.0, 20.0, 30.0}, 0.0);

            assertEquals(10.0, detection.get(0), EPSILON);
            assertEquals(20.0, detection.get(1), EPSILON);
            assertEquals(30.0, detection.get(2), EPSILON);
        }
    }

    @Nested
    @DisplayName("Metadata")
    class Metadata {

        @Test
        @DisplayName("stores and retrieves metadata")
        void storesAndRetrievesMetadata() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);
            detection.setMetadata("sensor-1");

            assertEquals("sensor-1", detection.getMetadata());
        }

        @Test
        @DisplayName("returns null for unset metadata")
        void returnsNullForUnsetMetadata() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);
            assertNull(detection.getMetadata());
        }

        @Test
        @DisplayName("allows any metadata type")
        void allowsAnyMetadataType() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);

            detection.setMetadata(42);
            assertEquals(42, detection.getMetadata());

            detection.setMetadata(new Object());
            assertNotNull(detection.getMetadata());
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("equals based on measurement and timestamp")
        void equalsBasedOnMeasurementAndTimestamp() {
            Detection d1 = Detection.of(new double[]{1.0, 2.0}, 0.5);
            Detection d2 = Detection.of(new double[]{1.0, 2.0}, 0.5);
            Detection d3 = Detection.of(new double[]{1.0, 3.0}, 0.5);
            Detection d4 = Detection.of(new double[]{1.0, 2.0}, 1.0);

            assertEquals(d1, d2);
            assertNotEquals(d1, d3);  // Different measurement
            assertNotEquals(d1, d4);  // Different timestamp
        }

        @Test
        @DisplayName("equals handles null and other types")
        void equalsHandlesNullAndOtherTypes() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);

            assertNotEquals(detection, null);
            assertNotEquals(detection, "string");
        }

        @Test
        @DisplayName("equals handles same reference")
        void equalsHandlesSameReference() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);
            assertEquals(detection, detection);
        }

        @Test
        @DisplayName("hashCode consistent with equals")
        void hashCodeConsistentWithEquals() {
            Detection d1 = Detection.of(new double[]{1.0, 2.0}, 0.5);
            Detection d2 = Detection.of(new double[]{1.0, 2.0}, 0.5);

            assertEquals(d1.hashCode(), d2.hashCode());
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTest {

        @Test
        @DisplayName("produces readable output")
        void producesReadableOutput() {
            Detection detection = Detection.of(new double[]{100.0, 200.0}, 1.5);

            String str = detection.toString();
            assertTrue(str.contains("measurement"));
            assertTrue(str.contains("timestamp"));
            assertTrue(str.contains("1.5"));
        }
    }
}
