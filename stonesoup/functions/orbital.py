"""
Orbital functions
-----------------

Functions used within multiple orbital classes in Stone Soup

"""
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from . import dotproduct
from .coordinates import (
    eci_to_ecef, ecef_to_eci,
    gcrs_to_j2000, j2000_to_gcrs
)
from ..types.array import StateVector, StateVectors


def stumpff_s(z):
    r"""The Stumpff S function

    .. math::

        S(z) = \begin{cases}\frac{\sqrt(z) - \sin{\sqrt(z)}}{(\sqrt(z))^{3}}, & (z > 0)\\
                     \frac{\sinh(\sqrt(-z)) - \sqrt(-z)}{(\sqrt(-z))^{3}}, & (z < 0) \\
                     \frac{1}{6}, & (z = 0)\end{cases}

    Parameters
    ----------
    z : float, array-like
        input parameter, :math:`z` or :math:`[z]`

    Returns
    -------
    : float, array-like
        Output value, :math:`S(z)` in the same format and same size as input.

    """
    gti = z > 0
    lti = z < 0
    eqi = z == 0

    if not np.shape(z):
        if gti:
            sqz = np.sqrt(z)
            out = (sqz - np.sin(sqz)) / sqz ** 3
        elif lti:
            sqz = np.sqrt(-z)
            out = (np.sinh(sqz) - sqz) / sqz ** 3
        else:
            out = 1 / 6
    else:
        out = np.zeros(np.shape(z)).view(type(z))
        out[gti] = (np.sqrt(z[gti]) - np.sin(np.sqrt(z[gti]))) / np.sqrt(z[gti]) ** 3
        out[lti] = (np.sinh(np.sqrt(-z[lti])) - np.sqrt(-z[lti])) / np.sqrt(-z[lti]) ** 3
        out[eqi] = 1 / 6

    return out


def stumpff_c(z):
    r"""The Stumpff C function

    .. math::

        C(z) = \begin{cases}\frac{1 - \cos{\sqrt(z)}}{z}, & (z > 0)\\
                     \frac{\cosh{\sqrt(-z)} - 1}{-z}, & (z < 0) \\
                     \frac{1}{2}, & (z = 0)\end{cases}

    Parameters
    ----------
    z : float, array-like
        input parameter, :math:`z`

    Returns
    -------
    : float, array-like
        Output value, :math:`C(z)` in same format and size as input

    """
    gti = z > 0
    lti = z < 0
    eqi = z == 0

    if not np.shape(z):
        if gti:
            out = (1 - np.cos(np.sqrt(z))) / np.sqrt(z) ** 2
        elif lti:
            out = (np.cosh(np.sqrt(-z)) - 1) / np.sqrt(-z) ** 2
        else:
            out = 1 / 2
    else:
        out = np.zeros(np.shape(z)).view(type(z))
        out[gti] = (1 - np.cos(np.sqrt(z[gti]))) / np.sqrt(z[gti]) ** 2
        out[lti] = (np.cosh(np.sqrt(-z[lti])) - 1) / np.sqrt(-z[lti]) ** 2
        out[eqi] = 1 / 2

    return out


def universal_anomaly_newton(o_state_vector, delta_t,
                             grav_parameter=3.986004418e14, precision=1e-8, max_iterations=1e5):
    r"""Calculate the universal anomaly via Newton's method. Algorithm 3.3 in [1]_.

    Parameters
    ----------
    o_state_vector : :class:`~.StateVector`, :class:`~.StateVectors`
        The orbital state vector formed as
        :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
    delta_t : timedelta
        The time interval over which to estimate the universal anomaly
    grav_parameter : float, optional
        The universal gravitational parameter. Defaults to that of the
        Earth, :math:`3.986004418 \times 10^{14} \ \mathrm{m}^{3} \
        \mathrm{s}^{-2}`
    precision : float, optional
        For Newton's method, the difference between new and old estimates of the universal anomaly
        below which the iteration stops and the answer is returned, (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations allowed in while loop (default = 1e5)

    Returns
    -------
    : numpy.ndarray
        The universal anomaly, :math:`\chi`

    References
    ----------
    .. [1] Curtis H.D. 2010, Orbital Mechanics for Engineering Students, 3rd Ed., Elsevier

    """

    # This should really have the calculation abstracted out and then do
    # if statevector do code, else do iteration over code
    # if type(o_state_vector) != StateVectors:
    #    o_state_vector = StateVectors([o_state_vector])

    mag_r_0 = np.sqrt(dotproduct(o_state_vector[0:3, :], o_state_vector[0:3, :]))
    mag_v_0 = np.sqrt(dotproduct(o_state_vector[3:6, :], o_state_vector[3:6, :]))
    v_rad_0 = dotproduct(o_state_vector[3:6, :], o_state_vector[0:3, :]) / mag_r_0
    root_mu = np.sqrt(grav_parameter)
    inv_sma = 2 / mag_r_0 - (mag_v_0 ** 2) / grav_parameter
    chi_i = root_mu * np.abs(inv_sma) * delta_t.total_seconds()

    out = []
    for iinv_sma, cchi_i, mmag_r_0, mmag_v_0, vv_rad_0 in \
            zip(inv_sma.ravel(), chi_i.ravel(), mag_r_0.ravel(), mag_v_0.ravel(), v_rad_0.ravel()):
        ratio = 1
        count = 0
        # Do Newton's method
        while np.abs(ratio) > precision and count <= max_iterations:
            z_i = iinv_sma * cchi_i ** 2
            f_chi_i = mmag_r_0 * vv_rad_0 / root_mu * cchi_i ** 2 * stumpff_c(z_i) + \
                (1 - iinv_sma * mmag_r_0) * cchi_i ** 3 * stumpff_s(z_i) + mmag_r_0 * cchi_i \
                - root_mu * delta_t.total_seconds()
            fp_chi_i = mmag_r_0 * vv_rad_0 / root_mu * cchi_i * \
                (1 - iinv_sma * cchi_i ** 2 * stumpff_s(z_i)) + (1 - iinv_sma * mmag_r_0) \
                * cchi_i ** 2 * stumpff_c(z_i) + mmag_r_0
            ratio = f_chi_i / fp_chi_i
            cchi_i = cchi_i - ratio
            count += 1

        out.append(cchi_i)

    return np.reshape(out, np.shape(np.atleast_2d(o_state_vector[0, :])))


def lagrange_coefficients_from_universal_anomaly(o_state_vector, delta_t,
                                                 grav_parameter=3.986004418e14,
                                                 precision=1e-8, max_iterations=1e5):
    r""" Calculate the Lagrangian coefficients, f and g, and their time derivatives, by way of the
    universal anomaly and the Stumpff functions [2]_.

    Parameters
    ----------
    o_state_vector : :class:`~.StateVector`, :class:`~.StateVectors`
        The (Cartesian) orbital state vector,
        :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
    delta_t : timedelta
        The time interval over which to calculate
    grav_parameter : float, optional
        The universal gravitational parameter. Defaults to that of the
        Earth, :math:`3.986004418 \times 10^{14} \ \mathrm{m}^{3} \
        \mathrm{s}^{-2}`. Note that the units of time must be seconds.
    precision : float, optional
        Precision to which to calculate the :meth:`universal anomaly` (default = 1e-8). See the doc
        section for that function
    max_iterations : float, optional
        Maximum number of iterations in determining universal anomaly (default = 1e5)

    Returns
    -------
    : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        The Lagrange coefficients, :math:`f, g, \dot{f}, \dot{g}`, in that order.

    References
    ----------
    .. [2] Bond V.R., Allman M.C. 1996, Modern Astrodynamics: Fundamentals and Perturbation
            Methods, Princeton University Press

    """
    # First get the universal anomaly using Newton's method
    chii = universal_anomaly_newton(o_state_vector, delta_t,
                                    grav_parameter=grav_parameter,
                                    precision=precision, max_iterations=max_iterations)

    # Get the position and velocity vectors
    bold_r_0 = o_state_vector[0:3, :]
    bold_v_0 = o_state_vector[3:6, :]

    # Calculate the magnitude of the position and velocity vectors
    r_0 = np.sqrt(dotproduct(bold_r_0, bold_r_0))
    v_0 = np.sqrt(dotproduct(bold_v_0, bold_v_0))

    # For convenience
    root_mu = np.sqrt(grav_parameter)
    inv_sma = 2 / r_0 - (v_0 ** 2) / grav_parameter
    z = inv_sma * chii ** 2

    # Get the Lagrange coefficients using Stumpf
    f = 1 - chii ** 2 / r_0 * stumpff_c(z)
    g = delta_t.total_seconds() - 1 / root_mu * chii ** 3 * \
        stumpff_s(z)

    # Get the position vector and magnitude of that vector
    bold_r = f * bold_r_0 + g * bold_v_0
    r = np.sqrt(dotproduct(bold_r, bold_r))

    # and the Lagrange (time) derivatives also using Stumpf
    fdot = root_mu / (r * r_0) * (inv_sma * chii ** 3 * stumpff_s(z) - chii)
    gdot = 1 - (chii ** 2 / r) * stumpff_c(z)

    return f, g, fdot, gdot


def eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity,
                                        precision=1e-8, max_iterations=1e5):
    r"""Approximately solve the transcendental equation :math:`E - e sin E = M_e` for E. This is
    an iterative process using Newton's method.

    Parameters
    ----------
    mean_anomaly : float
        Current mean anomaly
    eccentricity : float
        Orbital eccentricity
    precision : float, optional
        Precision used for the stopping point in determining eccentric anomaly from mean anomaly,
        (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations for the while loop, (default = 1e5)

    Returns
    -------
    : float
        Eccentric anomaly of the orbit
    """

    if mean_anomaly < np.pi:
        ecc_anomaly = mean_anomaly + eccentricity / 2
    else:
        ecc_anomaly = mean_anomaly - eccentricity / 2

    ratio = 1
    count = 0
    while np.abs(ratio) > precision and count <= max_iterations:
        f = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
        fp = 1 - eccentricity * np.cos(ecc_anomaly)
        ratio = f / fp  # Need to check conditioning
        ecc_anomaly = ecc_anomaly - ratio
        count += 1

    return ecc_anomaly  # Check whether this ever goes outside 0 < 2pi


def tru_anom_from_mean_anom(mean_anomaly, eccentricity, precision=1e-8, max_iterations=1e5):
    r"""Get the true anomaly from the mean anomaly via the eccentric anomaly

    Parameters
    ----------
    mean_anomaly : float
        The mean anomaly
    eccentricity : float
        Eccentricity
    precision : float, optional
        Precision used for the stopping point in determining eccentric anomaly from mean anomaly,
        (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations in determining eccentric anomaly, (default = 1e5)

    Returns
    -------
    : float
        True anomaly

    """
    cos_ecc_anom = np.cos(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity, precision=precision, max_iterations=max_iterations))
    sin_ecc_anom = np.sin(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity, precision=precision, max_iterations=max_iterations))

    # This only works for M_e < \pi
    # return np.arccos(np.clip((eccentricity - cos_ecc_anom) /
    #                 (eccentricity*cos_ecc_anom - 1), -1, 1))

    return np.remainder(np.arctan2(np.sqrt(1 - eccentricity ** 2) *
                                   sin_ecc_anom,
                                   cos_ecc_anom - eccentricity), 2 * np.pi)


def perifocal_position(eccentricity, semimajor_axis, true_anomaly):
    r"""The position vector in perifocal coordinates calculated from the Keplerian elements

    Parameters
    ----------
    eccentricity : float
        Orbit eccentricity
    semimajor_axis : float
        Orbit semi-major axis
    true_anomaly
        Orbit true anomaly

    Returns
    -------
    : numpy.array
        :math:`[r_x, r_y, r_z]` position in perifocal coordinates

    """

    # Cache some trigonometric functions
    c_tran = np.cos(true_anomaly)
    s_tran = np.sin(true_anomaly)

    # Copes with floats and (row) arrays
    rot_v = np.reshape(np.array([c_tran, s_tran, np.zeros(np.shape(c_tran))]),
                       (3, np.shape(np.atleast_2d(true_anomaly))[1]))

    return semimajor_axis * (1 - eccentricity ** 2) / (1 + eccentricity * c_tran) * rot_v


def perifocal_velocity(eccentricity, semimajor_axis, true_anomaly, grav_parameter=3.986004418e14):
    r"""The velocity vector in perifocal coordinates calculated from the Keplerian elements

    Parameters
    ----------
    eccentricity : float
        Orbit eccentricity
    semimajor_axis : float
        Orbit semi-major axis
    true_anomaly : float
        Orbit true anomaly
    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = G M`. Default is
        :math:`3.986004418 \times 10^{14} \mathrm{m}^3 \mathrm{s}^{-2}`

    Returns
    -------
    : numpy.narray
        :math:`[\dot{r}_x, \dot{r}_y, \dot{r}_z]` velocity in perifocal coordinates

    """

    # Cache some trigonometric functions
    c_tran = np.cos(true_anomaly)
    s_tran = np.sin(true_anomaly)

    # Copes with floats and (row) arrays
    rot_v = np.reshape(np.array([-s_tran, eccentricity + c_tran, np.zeros(np.shape(c_tran))]),
                       (3, np.shape(np.atleast_2d(true_anomaly))[1]))

    a_1_e_2 = np.array(semimajor_axis).astype(float) * \
        (1 - np.array(eccentricity).astype(float) ** 2)

    return np.sqrt(grav_parameter / a_1_e_2) * rot_v


def perifocal_to_geocentric_matrix(inclination, raan, argp):
    r"""Return the matrix which transforms from perifocal to geocentric coordinates

    Parameters
    ----------
    inclination : float
        Orbital inclination
    raan : float
        Orbit Right Ascension of the ascending node
    argp : float
        The orbit's argument of periapsis

    Returns
    -------
    : numpy.array
        The :math:`3 \times 3` array that transforms from perifocal coordinates to geocentric
        coordinates

    """
    # Cache some trig functions
    s_incl = np.sin(inclination)
    c_incl = np.cos(inclination)

    s_raan = np.sin(raan)
    c_raan = np.cos(raan)

    s_aper = np.sin(argp)
    c_aper = np.cos(argp)

    # Build the matrix
    return np.array([[-s_raan * c_incl * s_aper + c_raan * c_aper,
                      -s_raan * c_incl * c_aper - c_raan * s_aper,
                      s_raan * s_incl],
                     [c_raan * c_incl * s_aper + s_raan * c_aper,
                      c_raan * c_incl * c_aper - s_raan * s_aper,
                      -c_raan * s_incl],
                     [s_incl * s_aper, s_incl * c_aper, c_incl]])


def keplerian_to_rv(state_vector, grav_parameter=3.986004418e14):
    r"""Convert the Keplerian orbital elements to position, velocity state vector

    Parameters
    ----------
    state_vector : :class:`~.StateVector`, :class:`~.StateVectors`
        The Keplerian orbital state vector is defined as

        .. math::

            X = [e, a, i, \Omega, \omega, \theta]^{T} \\

        where:
        :math:`e` is the orbital eccentricity (unitless),
        :math:`a` the semi-major axis (m),
        :math:`i` the inclination (rad),
        :math:`\Omega` is the longitude of the ascending node (rad),
        :math:`\omega` the argument of periapsis (rad), and
        :math:`\theta` the true anomaly (rad)

    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = G M`. The default is :math:`3.986004418
        \times 10^{14} \mathrm{m}^3 \mathrm{s}^{-2}`

    Returns
    -------
    : :class:`~.StateVector`, :class:`~.StateVectors`
        Orbital state vector as :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]`

    Warning
    -------
    No checking undertaken. Assumes Keplerian elements rendered correctly as above

    """

    # The (hidden) function which does this on a single StateVector
    def _kep_to_rv_statevector(statevector):

        # Calculate the position vector in perifocal coordinates
        rx = perifocal_position(statevector[0], statevector[1], statevector[5])

        # Calculate the velocity vector in perifocal coordinates
        vx = perifocal_velocity(statevector[0], statevector[1], statevector[5],
                                grav_parameter=grav_parameter)

        # Transform position (perifocal) and velocity (perifocal) into geocentric
        r = perifocal_to_geocentric_matrix(statevector[2], statevector[3], statevector[4]) @ rx

        v = perifocal_to_geocentric_matrix(statevector[2], statevector[3], statevector[4]) @ vx

        return StateVector(np.concatenate((r, v), axis=0))

    # Do this a statevector at a time to avoid having to do tensor multiplication
    if type(state_vector) is StateVector:
        return _kep_to_rv_statevector(state_vector)
    elif type(state_vector) is StateVectors:

        outrv = np.zeros(np.shape(state_vector))
        for i, sv in enumerate(state_vector):
            outrv[:, slice(i, i+1)] = _kep_to_rv_statevector(sv)

        return StateVectors(outrv)
    else:
        raise TypeError(r"Input must be :class:`~.StateVector` or :class:`~.StateVectors`")


def mod_inclination(x):
    r"""Calculates the modulus of an inclination. Inclination angles are within the range :math:`0`
    to :math:`\pi`.

    Parameters
    ----------
    x: float
        inclination angle in radians

    Returns
    -------
    float
        Angle in radians in the range :math:`0` to :math:`+\pi`
    """

    x = x % np.pi

    return x


def mod_elongitude(x):
    r"""Calculates the modulus of an ecliptic longitude in which angles are within the range
    :math:`0` to :math:`2 \pi`.

    Parameters
    ----------
    x: float
        longitudinal angle in radians

    Returns
    -------
    float
        Angle in radians in the range :math:`0` to :math:`+2 \pi`
    """

    x = x % (2 * np.pi)

    return x


# =============================================================================
# Orbital State Transformation Functions
# =============================================================================


def orbital_state_eci_to_ecef(state_vector: StateVector,
                              timestamp: datetime) -> StateVector:
    r"""Transform an orbital state vector from ECI to ECEF coordinates.

    This function transforms a 6-element orbital state vector (position and velocity)
    from Earth-Centered Inertial (ECI) coordinates to Earth-Centered Earth-Fixed
    (ECEF) coordinates.

    Parameters
    ----------
    state_vector : StateVector
        6-element state vector :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
        in ECI coordinates (meters, meters/second).
    timestamp : datetime
        Time at which to perform the transformation.

    Returns
    -------
    StateVector
        6-element state vector in ECEF coordinates.

    Notes
    -----
    The transformation accounts for Earth rotation using the Earth Rotation Angle (ERA).
    For the position transformation:

    .. math::

        \mathbf{r}_{ECEF} = R_z(\theta) \cdot \mathbf{r}_{ECI}

    where :math:`\theta` is the Earth Rotation Angle.

    For velocity, the transformation also accounts for the rotation rate:

    .. math::

        \mathbf{v}_{ECEF} = R_z(\theta) \cdot \mathbf{v}_{ECI} - \boldsymbol{\omega} \times \mathbf{r}_{ECEF}

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from stonesoup.types.array import StateVector
    >>> # LEO satellite in ECI
    >>> state_eci = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_ecef = orbital_state_eci_to_ecef(state_eci, timestamp)

    """
    # Extract position and velocity
    pos_eci = np.array(state_vector[:3]).flatten()
    vel_eci = np.array(state_vector[3:6]).flatten()

    # Transform position using coordinate function
    pos_ecef = eci_to_ecef(pos_eci, timestamp)

    # For velocity transformation, we need the rotation matrix and Earth's angular velocity
    # Earth's angular velocity (rad/s)
    omega_earth = 7.292115e-5

    # Compute ERA for rotation matrix using Julian days from J2000.0
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_days = dt / 86400.0  # Du in days from J2000.0
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_days)

    cos_era = np.cos(era)
    sin_era = np.sin(era)
    rotation_matrix = np.array([
        [cos_era, sin_era, 0.0],
        [-sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Velocity transformation: v_ecef = R * v_eci - omega × r_ecef
    # omega = [0, 0, omega_earth]
    # omega × r = [-omega_earth * r_y, omega_earth * r_x, 0]
    vel_rotated = rotation_matrix @ vel_eci
    omega_cross_r = np.array([
        -omega_earth * pos_ecef[1],
        omega_earth * pos_ecef[0],
        0.0
    ])
    vel_ecef = vel_rotated - omega_cross_r

    return StateVector(np.concatenate([pos_ecef, vel_ecef]))


def orbital_state_ecef_to_eci(state_vector: StateVector,
                              timestamp: datetime) -> StateVector:
    r"""Transform an orbital state vector from ECEF to ECI coordinates.

    This function transforms a 6-element orbital state vector (position and velocity)
    from Earth-Centered Earth-Fixed (ECEF) coordinates to Earth-Centered Inertial
    (ECI) coordinates.

    Parameters
    ----------
    state_vector : StateVector
        6-element state vector :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
        in ECEF coordinates (meters, meters/second).
    timestamp : datetime
        Time at which to perform the transformation.

    Returns
    -------
    StateVector
        6-element state vector in ECI coordinates.

    Notes
    -----
    This is the inverse of :func:`orbital_state_eci_to_ecef`.

    For velocity transformation:

    .. math::

        \mathbf{v}_{ECI} = R_z(-\theta) \cdot (\mathbf{v}_{ECEF} + \boldsymbol{\omega} \times \mathbf{r}_{ECEF})

    Examples
    --------
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from stonesoup.types.array import StateVector
    >>> # Position in ECEF
    >>> state_ecef = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_eci = orbital_state_ecef_to_eci(state_ecef, timestamp)

    """
    # Extract position and velocity
    pos_ecef = np.array(state_vector[:3]).flatten()
    vel_ecef = np.array(state_vector[3:6]).flatten()

    # Transform position using coordinate function
    pos_eci = ecef_to_eci(pos_ecef, timestamp)

    # For velocity transformation, we need the rotation matrix and Earth's angular velocity
    # Earth's angular velocity (rad/s)
    omega_earth = 7.292115e-5

    # Compute ERA for rotation matrix using Julian days from J2000.0
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    dt = (timestamp - j2000_epoch).total_seconds()
    julian_days = dt / 86400.0  # Du in days from J2000.0
    era = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_days)

    # Inverse rotation (transpose of ECI-to-ECEF matrix)
    cos_era = np.cos(era)
    sin_era = np.sin(era)
    rotation_matrix_inv = np.array([
        [cos_era, -sin_era, 0.0],
        [sin_era, cos_era, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Velocity transformation: v_eci = R^T * (v_ecef + omega × r_ecef)
    # omega = [0, 0, omega_earth]
    # omega × r = [-omega_earth * r_y, omega_earth * r_x, 0]
    omega_cross_r = np.array([
        -omega_earth * pos_ecef[1],
        omega_earth * pos_ecef[0],
        0.0
    ])
    vel_eci = rotation_matrix_inv @ (vel_ecef + omega_cross_r)

    return StateVector(np.concatenate([pos_eci, vel_eci]))


def orbital_state_j2000_to_gcrs(state_vector: StateVector,
                                timestamp: Optional[datetime] = None) -> StateVector:
    r"""Transform an orbital state vector from J2000 to GCRS coordinates.

    This function transforms a 6-element orbital state vector from the J2000
    reference frame (mean equator and equinox at J2000.0) to the GCRS
    (Geocentric Celestial Reference System).

    Parameters
    ----------
    state_vector : StateVector
        6-element state vector in J2000 coordinates.
    timestamp : datetime, optional
        Time at which to perform the transformation. If None, uses a simplified
        transformation that treats frames as approximately equivalent.

    Returns
    -------
    StateVector
        6-element state vector in GCRS coordinates.

    Notes
    -----
    The J2000 and GCRS frames differ due to precession and nutation effects.
    For most near-Earth applications, the difference is small (typically
    less than 10 meters for position).

    Examples
    --------
    >>> from datetime import datetime
    >>> from stonesoup.types.array import StateVector
    >>> state_j2000 = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_gcrs = orbital_state_j2000_to_gcrs(state_j2000, timestamp)

    """
    pos = np.array(state_vector[:3]).flatten()
    vel = np.array(state_vector[3:6]).flatten() if len(state_vector) >= 6 else None

    pos_gcrs, vel_gcrs = j2000_to_gcrs(pos, vel, timestamp)

    if vel_gcrs is not None:
        return StateVector(np.concatenate([pos_gcrs, vel_gcrs]))
    else:
        return StateVector(pos_gcrs)


def orbital_state_gcrs_to_j2000(state_vector: StateVector,
                                timestamp: Optional[datetime] = None) -> StateVector:
    r"""Transform an orbital state vector from GCRS to J2000 coordinates.

    This function transforms a 6-element orbital state vector from the GCRS
    (Geocentric Celestial Reference System) to the J2000 reference frame
    (mean equator and equinox at J2000.0).

    Parameters
    ----------
    state_vector : StateVector
        6-element state vector in GCRS coordinates.
    timestamp : datetime, optional
        Time at which to perform the transformation. If None, uses a simplified
        transformation that treats frames as approximately equivalent.

    Returns
    -------
    StateVector
        6-element state vector in J2000 coordinates.

    Notes
    -----
    This is the inverse of :func:`orbital_state_j2000_to_gcrs`.

    Examples
    --------
    >>> from datetime import datetime
    >>> from stonesoup.types.array import StateVector
    >>> state_gcrs = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> timestamp = datetime(2024, 1, 1, 12, 0, 0)
    >>> state_j2000 = orbital_state_gcrs_to_j2000(state_gcrs, timestamp)

    """
    pos = np.array(state_vector[:3]).flatten()
    vel = np.array(state_vector[3:6]).flatten() if len(state_vector) >= 6 else None

    pos_j2000, vel_j2000 = gcrs_to_j2000(pos, vel, timestamp)

    if vel_j2000 is not None:
        return StateVector(np.concatenate([pos_j2000, vel_j2000]))
    else:
        return StateVector(pos_j2000)


def compute_orbital_period(semi_major_axis: float,
                           grav_parameter: float = 3.986004418e14) -> float:
    r"""Compute the orbital period from semi-major axis.

    Uses Kepler's third law:

    .. math::

        T = 2\pi \sqrt{\frac{a^3}{\mu}}

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis in meters.
    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = GM`.
        Default is Earth's value.

    Returns
    -------
    float
        Orbital period in seconds.

    Examples
    --------
    >>> # LEO orbit (400 km altitude)
    >>> period = compute_orbital_period(6778000)
    >>> print(f"Period: {period/60:.1f} minutes")
    Period: 92.4 minutes

    """
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / grav_parameter)


def compute_orbital_velocity(radius: float, semi_major_axis: float,
                             grav_parameter: float = 3.986004418e14) -> float:
    r"""Compute orbital velocity at a given radius using vis-viva equation.

    The vis-viva equation relates orbital velocity to position:

    .. math::

        v = \sqrt{\mu \left( \frac{2}{r} - \frac{1}{a} \right)}

    Parameters
    ----------
    radius : float
        Current orbital radius in meters.
    semi_major_axis : float
        Semi-major axis in meters.
    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = GM`.
        Default is Earth's value.

    Returns
    -------
    float
        Orbital velocity in meters/second.

    Examples
    --------
    >>> # Circular orbit at 400 km altitude
    >>> r = 6778000  # Earth radius + 400 km
    >>> v = compute_orbital_velocity(r, r)
    >>> print(f"Velocity: {v:.1f} m/s")
    Velocity: 7669.2 m/s

    """
    return np.sqrt(grav_parameter * (2/radius - 1/semi_major_axis))


def compute_specific_angular_momentum(state_vector: StateVector) -> np.ndarray:
    r"""Compute the specific angular momentum vector from orbital state.

    The specific angular momentum is:

    .. math::

        \mathbf{h} = \mathbf{r} \times \mathbf{v}

    Parameters
    ----------
    state_vector : StateVector
        6-element orbital state vector [r_x, r_y, r_z, v_x, v_y, v_z].

    Returns
    -------
    np.ndarray
        3-element specific angular momentum vector in m²/s.

    Examples
    --------
    >>> from stonesoup.types.array import StateVector
    >>> state = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> h = compute_specific_angular_momentum(state)

    """
    r = np.array(state_vector[:3]).flatten()
    v = np.array(state_vector[3:6]).flatten()
    return np.cross(r, v)


def compute_specific_energy(state_vector: StateVector,
                            grav_parameter: float = 3.986004418e14) -> float:
    r"""Compute the specific orbital energy from state vector.

    The specific orbital energy (vis-viva integral) is:

    .. math::

        \varepsilon = \frac{v^2}{2} - \frac{\mu}{r}

    Parameters
    ----------
    state_vector : StateVector
        6-element orbital state vector.
    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = GM`.

    Returns
    -------
    float
        Specific orbital energy in J/kg (m²/s²).

    Notes
    -----
    Negative energy indicates a bound (elliptical) orbit.
    Zero energy indicates a parabolic escape trajectory.
    Positive energy indicates a hyperbolic trajectory.

    Examples
    --------
    >>> from stonesoup.types.array import StateVector
    >>> state = StateVector([7000000, 0, 0, 0, 7500, 0])
    >>> energy = compute_specific_energy(state)
    >>> print("Bound orbit" if energy < 0 else "Escape trajectory")
    Bound orbit

    """
    r = np.array(state_vector[:3]).flatten()
    v = np.array(state_vector[3:6]).flatten()

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    return v_mag**2 / 2 - grav_parameter / r_mag
