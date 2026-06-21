import numpy as np


def generate_radiation_pattern(
    phases_deg,
    amplitudes=None,
    num_elements=4,
    d=0.5,
    theta_points=181
):
    """
    Generate normalized radiation pattern for a 4-element linear array.

    Parameters
    ----------
    phases_deg : list
        Phase values in degrees.
        Example: [0, 20, 40, 60]

    amplitudes : list
        Amplitudes of each element.
        Default = [1,1,1,1]

    Returns
    -------
    theta_deg : ndarray
        Angles from 0° to 180°

    pattern : ndarray
        Normalized radiation pattern
    """

    if amplitudes is None:
        amplitudes = np.ones(num_elements)

    theta = np.linspace(0, np.pi, theta_points)

    k = 2 * np.pi

    af = np.zeros(theta_points, dtype=complex)

    for n in range(num_elements):

        phase_rad = np.deg2rad(phases_deg[n])

        af += amplitudes[n] * np.exp(
            1j * (
                n * k * d * np.cos(theta)
                + phase_rad
            )
        )

    pattern = np.abs(af)

    pattern = pattern / np.max(pattern)

    theta_deg = np.rad2deg(theta)

    return theta_deg, pattern