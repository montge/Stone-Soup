Interface Requirements
======================

This document defines interface and integration requirements for Stone Soup.

Python API
----------

.. icd:: Python 3.10+ Compatibility
   :id: ICD-PY-001
   :status: implemented
   :tags: interface, python

   The system shall be compatible with Python 3.10 and later versions,
   following standard Python packaging conventions.

.. icd:: NumPy Integration
   :id: ICD-PY-002
   :status: implemented
   :tags: interface, python, numpy

   State vectors and covariance matrices shall be compatible with
   NumPy arrays, supporting standard NumPy operations and broadcasting.

.. icd:: Component Interoperability
   :id: ICD-PY-003
   :status: implemented
   :tags: interface, python, components

   All Stone Soup components shall follow the base class interfaces
   defined in stonesoup.base, enabling component interchangeability.

C Library API
-------------

.. icd:: C17 Standard Compliance
   :id: ICD-C-001
   :status: implemented
   :tags: interface, c

   The libstonesoup C library shall comply with the C17 standard
   for maximum portability.

.. icd:: Error Handling
   :id: ICD-C-002
   :status: implemented
   :tags: interface, c, error

   The C library shall use consistent error return codes
   (stonesoup_error_t) for all functions, with descriptive
   error messages available via stonesoup_error_string().

.. icd:: Memory Management
   :id: ICD-C-003
   :status: implemented
   :tags: interface, c, memory

   The C library shall provide explicit create/free functions
   for all dynamically allocated types, with no hidden allocations.

.. icd:: Thread Safety
   :id: ICD-C-004
   :status: partial
   :tags: interface, c, threading

   The C library shall be thread-safe for read operations.
   Write operations on shared data structures require external
   synchronization.

Data Formats
------------

.. icd:: Timestamp Support
   :id: ICD-DATA-001
   :status: implemented
   :tags: interface, data

   All state types shall support timestamps using Python's
   datetime objects, with optional timezone awareness.

.. icd:: Serialization Support
   :id: ICD-DATA-002
   :status: implemented
   :tags: interface, data, serialization

   Core types shall support serialization/deserialization
   using YAML format for configuration persistence.

External Integration
--------------------

.. icd:: Matplotlib Integration
   :id: ICD-EXT-001
   :status: implemented
   :tags: interface, visualization

   The system shall provide plotting utilities compatible with
   Matplotlib for visualizing tracks, detections, and metrics.

.. icd:: SciPy Integration
   :id: ICD-EXT-002
   :status: implemented
   :tags: interface, scipy

   Mathematical operations shall leverage SciPy where appropriate
   for optimized linear algebra and statistical functions.
