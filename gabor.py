import cv2
import numpy as np
from skimage.filters import gabor_kernel

N_ORIENT = 8
N_SCALE = 5
UL, UH = (0.05, 0.4)


def manjunath_ma_params(ul=UL, uh=UH, n_scale=N_SCALE, n_orient=N_ORIENT):
    """Return [(frequency, sigma_x, sigma_y)] per scale, finest scale first."""
    if not 0 < ul < uh < 0.5 or n_scale < 2 or n_orient < 2:
        raise ValueError(
            "Use 0 < ul < uh < 0.5 and at least two scales and orientations."
        )
    a = (uh / ul) ** (1.0 / (n_scale - 1))
    ln2 = np.log(2.0)
    su = (a - 1) * uh / ((a + 1) * np.sqrt(2 * ln2))
    sv = (
        np.tan(np.pi / (2 * n_orient))
        * (uh - 2 * ln2 * su**2 / uh)
        / np.sqrt(2 * ln2 - (2 * ln2) ** 2 * su**2 / uh**2)
    )
    sx, sy = (1 / (2 * np.pi * su), 1 / (2 * np.pi * sv))
    return [(uh / a**m, sx * a**m, sy * a**m) for m in range(n_scale)]


def orientations(n=N_ORIENT):
    """Feature (tangent) orientations theta_i = i*pi/n in [0, pi)."""
    return np.arange(n) * np.pi / n


def gabor_bank(gray, n_orient=N_ORIENT, n_scale=N_SCALE, ul=UL, uh=UH, n_stds=3):
    """Gabor response magnitudes summed over frequency: R[i] for tangent orientation theta_i."""
    gray = gray.astype(np.float32)
    gray -= float(np.mean(gray, dtype=np.float64))
    R = np.zeros((n_orient,) + gray.shape, np.float32)
    for i, theta in enumerate(orientations(n_orient)):
        for f, sx, sy in manjunath_ma_params(ul, uh, n_scale, n_orient):
            k = gabor_kernel(
                f, theta=theta + np.pi / 2, sigma_x=sx, sigma_y=sy, n_stds=n_stds
            )
            re = (k.real - k.real.mean()).astype(np.float32)
            im = (k.imag - k.imag.mean()).astype(np.float32)
            r = cv2.filter2D(gray, -1, re, borderType=cv2.BORDER_REFLECT)
            q = cv2.filter2D(gray, -1, im, borderType=cv2.BORDER_REFLECT)
            R[i] += np.sqrt(r * r + q * q)
    return R / max(R.max(), 1e-12)
