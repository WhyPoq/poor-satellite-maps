from math import comb

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from gabor import orientations

SIGMA = 12.0
N_ANGULAR = 8


def encode(R):
    """Eq 7: T = sum_i R_i t_i t_i^T with t_i the unit tangent of filter i. Returns (Txx, Txy, Tyy)."""
    th = orientations(len(R))[:, None, None]
    c, s = (np.cos(th), np.sin(th))
    return ((R * c * c).sum(0), (R * c * s).sum(0), (R * s * s).sum(0))


def decompose(Txx, Txy, Tyy):
    """2x2 tensor -> (stick saliency l1-l2, ball saliency l2, tangent angle of e1 in [0, pi))."""
    half_tr = (Txx + Tyy) / 2
    w = np.hypot((Txx - Tyy) / 2, Txy)
    l1, l2 = (half_tr + w, np.maximum(half_tr - w, 0))
    theta = np.mod(0.5 * np.arctan2(2 * Txy, Txx - Tyy), np.pi)
    theta[theta >= np.pi] = 0
    return (l1 - l2, l2, theta)


def _kernels(sigma, radius_factor=3):
    r = int(np.ceil(radius_factor * sigma / np.sqrt(2)))
    y, x = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    rad2 = x * x + y * y
    decay = np.exp(-rad2 / sigma**2)
    psi = np.arctan2(y, x)

    def harmonic(k):
        h = decay * np.exp(1j * k * psi)
        if k:
            h[r, r] = 0
        return h

    return harmonic


def vote(stick, ball, theta, sigma=SIGMA, n=N_ANGULAR, radius_factor=3):
    """Dense voting. Inputs per pixel: stick saliency, ball saliency, stick tangent angle."""
    if sigma <= 0 or n < 1 or radius_factor <= 0:
        raise ValueError("Use positive sigma, angular order and radius factor.")
    harmonic = _kernels(sigma, radius_factor)
    U = np.zeros(stick.shape, np.complex128)
    tr = np.zeros(stick.shape, np.float64)
    e2 = np.exp(-2j * theta)
    for j in range(-n, n + 1):
        c = comb(2 * n, n + j) / 4**n
        a = stick * e2 ** (j + 1)
        t = stick * e2**j
        if j == -1:
            a = a + 2 * ball
        if j == 0:
            t = t + 2 * ball
        U += c / 2 * fftconvolve(a, harmonic(2 * j + 4), mode="same")
        tr += c * fftconvolve(t, harmonic(2 * j), mode="same").real
    return (tr / 2 + U.real, U.imag, tr / 2 - U.real)


def curve_nms(sal, theta, floor=1e-3, distance=1):
    """Keep pixels whose saliency is a local max across the curve (along the normal of theta)."""
    h, w = sal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (-distance * np.sin(theta), distance * np.cos(theta))
    a = map_coordinates(sal, [yy + ny, xx + nx], order=1, mode="nearest")
    b = map_coordinates(sal, [yy - ny, xx - nx], order=1, mode="nearest")
    return (sal >= a) & (sal >= b) & (sal > floor * sal.max())


def infer(
    R,
    sigma=SIGMA,
    n=N_ANGULAR,
    floor=1e-3,
    distance=1,
    radius_factor=3,
    vote_balls=False,
):
    """Encode, vote and classify curve features; return intermediate maps."""
    encoded = encode(R)
    stick, ball, theta0 = decompose(*encoded)
    voted = vote(
        stick,
        ball if vote_balls else np.zeros_like(ball),
        theta0,
        sigma,
        n,
        radius_factor,
    )
    s, b, theta = decompose(*voted)
    is_curve = s > b
    pts = is_curve & curve_nms(s, theta, floor, distance)
    cmap = np.where(pts, s, 0) / max(s[pts].max() if pts.any() else 1, 1e-12)
    return {
        "encoded_xx": encoded[0],
        "encoded_xy": encoded[1],
        "encoded_yy": encoded[2],
        "voted_xx": voted[0],
        "voted_xy": voted[1],
        "voted_yy": voted[2],
        "stick": stick,
        "ball": ball,
        "initial_theta": theta0,
        "curve_sal": s,
        "junction_sal": b,
        "theta": theta,
        "is_curve": is_curve,
        "curve_points": pts,
        "curve_map": cmap.astype(np.float32),
    }
