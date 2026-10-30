import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

N_ORIENT = 8
WIDTHS = tuple(range(6, 41, 2))
SIGMA_SIDE, SIGMA_ALONG = (1.5, 4.0)


def make_segment(pts, theta, width, meta=None):
    """A polyline dictionary with per-vertex angles and widths."""
    return {
        "pts": np.asarray(pts, float),
        "theta": np.asarray(theta, float),
        "width": np.asarray(width, float),
        "meta": {} if meta is None else dict(meta),
    }


def segment_length(segment):
    return float(np.linalg.norm(np.diff(segment["pts"], axis=0), axis=1).sum())


def _gauss(x, y, cx, cy, sn, st, phi):
    """Anisotropic Gaussian at (cx, cy) in the filter frame: n across (sigma sn), t along (sigma st)."""
    c, s = (np.cos(phi), np.sin(phi))
    dx, dy = (x - cx, y - cy)
    along, across = (dx * c + dy * s, -dx * s + dy * c)
    return np.exp(-0.5 * (across**2 / sn**2 + along**2 / st**2)) / (2 * np.pi * sn * st)


def kernels(phi, w, sigma_side=SIGMA_SIDE, sigma_along=SIGMA_ALONG, area_scale=0.25):
    """Eq 20 bi-modal kernel (peaks w apart across a road of tangent phi) and the single-mode kernel."""
    if min(w, sigma_side, sigma_along, area_scale) <= 0:
        raise ValueError("Widths and Gaussian scales must be positive.")
    r = int(np.ceil(w / 2 + 3 * max(sigma_side, sigma_along, w * area_scale)))
    y, x = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    nx, ny = (-np.sin(phi), np.cos(phi))
    gb = _gauss(x, y, w / 2 * nx, w / 2 * ny, sigma_side, sigma_along, phi) + _gauss(
        x, y, -w / 2 * nx, -w / 2 * ny, sigma_side, sigma_along, phi
    )
    gs = _gauss(x, y, 0, 0, w * area_scale, sigma_along, phi)
    return (gb.astype(np.float32), (gs / gs.sum()).astype(np.float32))


def orientation_weight(theta, phi):
    return np.abs(np.cos(theta - phi)).astype(np.float32)


def filter_maps(
    curve_points,
    theta,
    road,
    n_orient=N_ORIENT,
    widths=WIDTHS,
    sigma_side=SIGMA_SIDE,
    sigma_along=SIGMA_ALONG,
    area_scale=0.25,
):
    """Gt = Gb * Gs over all (phi, w). Returns per-pixel argmax: M (magnitude), Phi, W (Fig 10a)."""
    road = np.asarray(road, dtype=bool)
    C = curve_points.astype(np.float32)
    mask = road.astype(np.float32)
    M = np.zeros(C.shape, np.float32)
    Phi = np.zeros(C.shape, np.float32)
    W = np.zeros(C.shape, np.float32)
    for phi in np.arange(n_orient) * np.pi / n_orient:
        Cw = C * orientation_weight(theta, phi)
        for w in widths:
            gb, gs = kernels(phi, w, sigma_side, sigma_along, area_scale)
            gt = cv2.filter2D(
                Cw, -1, gb, borderType=cv2.BORDER_CONSTANT
            ) * cv2.filter2D(mask, -1, gs, borderType=cv2.BORDER_CONSTANT)
            better = gt > M
            M[better], Phi[better], W[better] = (gt[better], phi, w)
    M[~road] = 0
    Phi[~road] = 0
    W[~road] = 0
    return (M, Phi, W)


def centerline_pixels(M, Phi, threshold=None, distance=1):
    """NMS across the road orientation, Otsu threshold on M, skeleton."""
    h, w = M.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (-distance * np.sin(Phi), distance * np.cos(Phi))
    a = map_coordinates(M, [yy + ny, xx + nx], order=1, mode="nearest")
    b = map_coordinates(M, [yy - ny, xx - nx], order=1, mode="nearest")
    ridge = (M >= a) & (M >= b) & (M > 0)
    if not ridge.any():
        return ridge
    cutoff = threshold_otsu(M[ridge]) if threshold is None else threshold
    return skeletonize(ridge & (M >= cutoff))


def iterative_hough(
    pix,
    M,
    Phi,
    W,
    n_orient=N_ORIENT,
    n_angle=180,
    line_distance=2.5,
    min_length_factor=2,
    gap_factor=1,
):
    """Convert centerline pixels into straight segment dictionaries."""
    if min(n_orient, n_angle, line_distance, min_length_factor, gap_factor) <= 0:
        raise ValueError("Hough parameters must be positive.")
    ys, xs = np.nonzero(pix)
    if len(xs) == 0:
        return []
    m, phi, wid = (M[ys, xs].astype(np.float64), Phi[ys, xs], W[ys, xs])
    med_w, med_m = (float(np.median(wid)), float(np.median(m)))
    min_run, gap = (min_length_factor * med_w, gap_factor * med_w)
    D = int(np.ceil(np.hypot(*M.shape)))
    angles = np.arange(n_angle) * np.pi / n_angle
    tol = round(n_angle / n_orient / 2)
    normal_bin = (
        np.round(np.mod(phi + np.pi / 2, np.pi) / np.pi * n_angle).astype(int) % n_angle
    )
    offs = np.arange(-tol, tol + 1)
    vk = np.mod(normal_bin[:, None] + offs[None], n_angle)
    vr = (
        np.round(
            xs[:, None] * np.cos(angles[vk]) + ys[:, None] * np.sin(angles[vk])
        ).astype(int)
        + D
    )
    acc = np.zeros((n_angle, 2 * D + 1))
    np.add.at(acc, (vk, vr), m[:, None])
    active = np.ones(len(xs), bool)
    segments = []
    stop = min_run * med_m
    while acc.max() > 0 and acc.max() >= stop:
        k, r = np.unravel_index(np.argmax(acc), acc.shape)
        t, rho = (angles[k], r - D)
        ct, st = (np.cos(t), np.sin(t))
        dist = np.abs(xs * ct + ys * st - rho)
        dk = np.abs(normal_bin - k)
        compat = active & (np.minimum(dk, n_angle - dk) <= tol)
        sel = np.nonzero(compat & (dist <= line_distance))[0]
        run = _best_run(sel, -xs * st + ys * ct, m, gap)
        s = -xs[run] * st + ys[run] * ct if len(run) else np.array([0.0])
        if len(run) and s.max() - s.min() >= min_run:
            w_run = float(np.median(wid[run]))
            ends = np.array(
                [[rho * ct - si * st, rho * st + si * ct] for si in (s.min(), s.max())]
            )
            theta = np.mod(t + np.pi / 2, np.pi)
            segments.append(
                make_segment(
                    ends,
                    np.full(2, theta),
                    np.full(2, w_run),
                    {"votes": float(acc[k, r])},
                )
            )
            along = -xs * st + ys * ct
            band = np.nonzero(
                compat
                & (dist <= w_run / 2)
                & (along >= s.min() - gap)
                & (along <= s.max() + gap)
            )[0]
            band = np.union1d(band, run)
            np.add.at(acc, (vk[band], vr[band]), -m[band, None])
            active[band] = False
        else:
            acc[max(k - 1, 0) : k + 2, max(r - 1, 0) : r + 2] = 0
    return segments


def _best_run(sel, along, m, gap):
    """Indices of the heaviest run of `sel` pixels along the line with no gap larger than `gap`."""
    if len(sel) == 0:
        return sel
    order = sel[np.argsort(along[sel])]
    breaks = np.nonzero(np.diff(along[order]) > gap)[0] + 1
    runs = np.split(order, breaks)
    return max(runs, key=lambda r: m[r].sum())


def extract(
    curve_points,
    theta,
    road,
    n_orient=N_ORIENT,
    widths=WIDTHS,
    sigma_side=SIGMA_SIDE,
    sigma_along=SIGMA_ALONG,
    area_scale=0.25,
    threshold=None,
    distance=1,
    n_angle=180,
    line_distance=2.5,
    min_length_factor=2,
    gap_factor=1,
):
    """Full Sec 6.1. Returns dict(M, Phi, W, pixels, segments)."""
    if n_orient < 1 or not len(widths) or min(widths) <= 0:
        raise ValueError("Use positive orientations and road widths.")
    M, Phi, W = filter_maps(
        curve_points, theta, road, n_orient, widths, sigma_side, sigma_along, area_scale
    )
    pix = centerline_pixels(M, Phi, threshold, distance)
    segments = iterative_hough(
        pix, M, Phi, W, n_orient, n_angle, line_distance, min_length_factor, gap_factor
    )
    return {"M": M, "Phi": Phi, "W": W, "pixels": pix, "segments": segments}
