"""Equal-area map of input directions for length-3 inputs.

A bias-free ReLU RNN is positively homogeneous, f(c x) = c f(x) for c > 0, so
its argmax depends only on the direction x / |x|. A standard Gaussian's
direction is uniform on the sphere, so accuracy = fraction of sphere area.

Coordinates (Archimedes' equal-area projection):
    axis e3 = (1, 1, 1) / sqrt(3)       z     = component along e3 ("mean" part)
    plane spanned by e1, e2             theta = angle in the plane ("ordering" part)
Uniform (theta, z) on [0, 2pi) x [-1, 1] is exactly uniform on the sphere, so
each pixel of the map has the same probability under Gaussian inputs.
The sorted order of (x1, x2, x3) depends only on theta: six 60-degree wedges.
"""

import numpy as np

E1 = np.array([1, -1, 0]) / np.sqrt(2)
E2 = np.array([1, 1, -2]) / np.sqrt(6)
E3 = np.array([1, 1, 1]) / np.sqrt(3)


def directions(theta, z):
    """Unit vectors (..., 3) for map coordinates theta, z (same shape)."""
    r = np.sqrt(1 - z**2)[..., None]
    plane = np.cos(theta)[..., None] * E1 + np.sin(theta)[..., None] * E2
    return (r * plane + z[..., None] * E3).astype(np.float32)


def grid(n_theta, n_z):
    """Pixel-centre grid. Returns theta, z of shape (n_z, n_theta)."""
    theta = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    z = (np.arange(n_z) + 0.5) / n_z * 2 - 1
    return np.meshgrid(theta, z)
