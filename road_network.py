"""Road network extraction and modelling (Poullis & You 2010, Section 6).

Input: the Stage 1 curve map and orientations (tensor voting), the Stage 2
road mask (graph cuts) and the BGR image. `extract` runs the automatic
network:

  1. Optional mask cleaning: oriented line openings keep long, straight
     road structures and drop blobs on roofs and in courtyards.
  2. Centreline filters G_b and G_s; for each pixel the maximum response and
     the orientation and width that gave it (Sec. 6.1, Eqs. 17-20).
  3. Iterative Hough transform with the responses as votes (Sec. 6.1),
     optionally keeping only the longest connected network (Sec. 7).
  4. Segments are cut where they leave the mask, near-parallel lane bands
     are merged into one centreline, and ends are extended to junctions.
  5. Road tracking: two road sides per centreline point (Sec. 6.2, Eq. 21)
     and the polygon union of the road segments.
  6. Optional refinement actions 1-7 (Sec. 6.2, Eqs. 22-27), then tracking
     again. The action format is documented in `refine`.

Coordinates are (x, y) image pixels, origin top left, y down.
"""
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
import shapely
import shapely.ops
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import LineString, mapping
from shapely.geometry.base import BaseGeometry

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


# ----------------------------------------------------------------------
# Sec. 6.1: centreline filters (Eqs. 17-20)
# ----------------------------------------------------------------------


@dataclass
class CentrelineResponse:
    """Per-pixel maximum filter response and the filter that produced it."""
    #  how strongly each pixel looks like a road centre. A big number means very likely. 0 means not at all
    magnitude: FloatArray 

    # radians in [0, pi), road direction. Which way the road runs at that pixel, as an angle.
    orientation: FloatArray

    # pixels, distance between the road sides
    width: FloatArray


def _rotated_grid(half: int, theta: float) -> tuple[FloatArray, FloatArray]:
    """Return (u, v): each kernel cell's distance along (u) and across (v) a road at angle theta.

    Rotating the coordinates makes every road look horizontal, so the kernels can put the
    road sides at v = ±w/2 for any angle (the "r" rotation in Eqs. 17–19).
    """
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    m = cv2.getRotationMatrix2D((0, 0), np.degrees(theta), 1.0)
    uv = np.asarray(cv2.transform(np.dstack([x, y]), m), dtype=np.float64)
    return uv[..., 0], uv[..., 1]


def _filter(image: FloatArray, kernel: FloatArray) -> FloatArray:
    """Filter with zero padding, output the same size as the image."""
    return np.asarray(cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT), dtype=np.float64)


def bimodal_kernel(width: float, theta: float, sigma_across: float, sigma_along: float) -> FloatArray:
    """Road-side filter G_b, Eq. (17): two Gaussians one road width apart.

    The exponent is as printed in the paper.
    """
    half = int(np.ceil(max(width / 2 + 3 * sigma_across, 3 * sigma_along)))
    u, v = _rotated_grid(half, theta)
    along = u ** 2 / sigma_along ** 2
    return (np.exp(-((v - width / 2) ** 2 / sigma_across ** 2 + along))
            + np.exp(-((v + width / 2) ** 2 / sigma_across ** 2 + along))
            ) / (2 * np.pi * sigma_across * sigma_along)


def single_mode_kernel(width: float, theta: float, sigma_along: float) -> FloatArray:
    """Road-area filter G_s: one Gaussian on the centreline.

    The paper only plots G_s (Fig. 9a). Assumption: sigma across = width / 4,
    so most of the kernel lies between the road sides. The kernel sums to 1,
    so its response on a binary road mask is the local fraction of road.
    """
    sigma_across = width / 4
    half = int(np.ceil(max(3 * sigma_across, 3 * sigma_along)))
    u, v = _rotated_grid(half, theta)
    k = np.exp(-(v ** 2 / sigma_across ** 2 + u ** 2 / sigma_along ** 2))
    kernel: FloatArray = k / k.sum()
    return kernel


def centreline_response(curve_saliency: ArrayLike, curve_orientation: ArrayLike, road_mask: ArrayLike,
                        widths: Sequence[float], n_orientations: int = 12,
                        sigma_across: float = 1.5, sigma_along: float = 6.0,
                        progress=None) -> CentrelineResponse:
    """Combined road-side and road-area response, G_t = G_b * G_s (Sec. 6.1).

    curve_saliency    Stage 1 curve likelihood (lambda2 - lambda3), 0 off curves.
    curve_orientation Stage 1 tangent orientation, radians in [0, pi).
    road_mask         Stage 2 binary road / non-road segmentation.
    widths            road widths to test, in pixels.

    Eq. (20) weights each road-side pixel by cos of the angle between its own
    orientation and the filter's. Orientations are undirected, so |cos| is used.
    That weighting is a per-pixel factor, so Eq. (20) is a convolution of the
    weighted saliency image with G_b.

    cv2.filter2D computes correlation; both kernels are symmetric about their
    centre, so that equals convolution. Pixels outside the image count as 0.

    The two responses are multiplied: a pair of parallel edges with no road
    between them (Fig. 9b, a building edge next to a road side) gets zero.
    """
    saliency = np.asarray(curve_saliency, dtype=np.float64)
    orientation = np.asarray(curve_orientation, dtype=np.float64)
    mask = np.asarray(road_mask, dtype=np.float64)
    best = np.zeros(saliency.shape)
    best_theta = np.zeros(saliency.shape)
    best_width = np.zeros(saliency.shape)
    for theta in np.arange(n_orientations) * np.pi / n_orientations:
        if progress:
            progress(f'centreline filters {np.degrees(theta):g}° / 180°')
        weighted = saliency * np.abs(np.cos(orientation - theta))
        for w in widths:
            r_b = _filter(weighted, bimodal_kernel(w, theta, sigma_across, sigma_along))
            r_s = _filter(mask, single_mode_kernel(w, theta, sigma_along))
            r = np.clip(r_b, 0, None) * np.clip(r_s, 0, 1)  # clip round-off
            better = r > best
            best[better] = r[better]
            best_theta[better] = theta
            best_width[better] = w
    return CentrelineResponse(best, best_theta, best_width)



def fixed_orientation_response(curve_saliency: ArrayLike, curve_orientation: ArrayLike, road_mask: ArrayLike,
                               points: ArrayLike, thetas: ArrayLike, widths: Sequence[float],
                               sigma_across: float = 1.5, sigma_along: float = 6.0) -> CentrelineResponse:
    """The same filters at given points only, each turned to a given orientation.

    Refinement action 2 (Sec. 6.2): along a centreline the user adds, the
    filters are applied at a fixed orientation aligned to its slope. Returns
    1-D arrays, one value per point: magnitude, the orientation used, and the
    width with the maximum response.
    """
    saliency = np.asarray(curve_saliency, dtype=np.float64)
    orientation = np.asarray(curve_orientation, dtype=np.float64)
    mask = np.asarray(road_mask, dtype=np.float64)
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(int)
    theta_at = np.mod(np.asarray(thetas, dtype=np.float64), np.pi)
    best = np.zeros(len(pts))
    best_width = np.zeros(len(pts))
    kernels: dict[tuple[float, float], tuple[FloatArray, FloatArray]] = {}
    for i, ((x, y), theta) in enumerate(zip(pts, theta_at)):
        theta = round(theta, 2)  # share kernels between points on one straight piece
        for w in widths:
            if (theta, w) not in kernels:
                kernels[theta, w] = (bimodal_kernel(w, theta, sigma_across, sigma_along),
                                     single_mode_kernel(w, theta, sigma_along))
            kb, ks = kernels[theta, w]
            weighted = _patch(saliency, x, y, kb.shape[0] // 2) * _patch_cos(orientation, x, y, kb.shape[0] // 2, theta)
            r_b = float((weighted * kb).sum())
            r_s = float((_patch(mask, x, y, ks.shape[0] // 2) * ks).sum())
            r = max(r_b, 0.0) * min(max(r_s, 0.0), 1.0)
            if r > best[i]:
                best[i], best_width[i] = r, w
    return CentrelineResponse(best, theta_at, best_width)


def _patch(image: FloatArray, x: int, y: int, half: int) -> FloatArray:
    """(2 half + 1) square window centred on (x, y), zero outside the image."""
    out = np.zeros((2 * half + 1, 2 * half + 1))
    h, w = image.shape
    r0, r1, c0, c1 = max(y - half, 0), min(y + half + 1, h), max(x - half, 0), min(x + half + 1, w)
    if r0 < r1 and c0 < c1:
        out[r0 - y + half:r1 - y + half, c0 - x + half:c1 - x + half] = image[r0:r1, c0:c1]
    return out


def _patch_cos(orientation: FloatArray, x: int, y: int, half: int, theta: float) -> FloatArray:
    """Eq. (20) weight |cos| of the angle between each pixel's orientation and theta, in a window."""
    return np.abs(np.cos(_patch(orientation, x, y, half) - theta))


# ----------------------------------------------------------------------
# Road sides from the Stage 2 mask
# ----------------------------------------------------------------------

def curves_from_mask(road_mask: BoolArray, sigma: float = 1.5) -> tuple[FloatArray, FloatArray]:
    """Road sides from the Stage 2 mask, in place of the Stage 1 curve map.

    Saliency 1 on the road sides (mask edge pixels), and their tangent.

    Returns (saliency, orientation). Orientation is the direction of the
    boundary, in [0, pi), from the gradient of the smoothed mask.
    """
    mask = road_mask.astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = road_mask & ~cv2.erode(mask, cross).astype(bool)  # road pixels with a non-road 4-neighbour
    size = 2 * round(4 * sigma) + 1
    smooth = cv2.GaussianBlur(road_mask.astype(np.float64), (size, size), sigma, borderType=cv2.BORDER_REFLECT)
    gy, gx = np.gradient(smooth)
    tangent = np.mod(np.arctan2(gy, gx) + np.pi / 2, np.pi)
    return edges.astype(np.float64), np.where(edges, tangent, 0.0)


# ----------------------------------------------------------------------
# Sec. 6.1: linearisation, iterative Hough transform (Fig. 10)
# ----------------------------------------------------------------------

N_ANGLES = 180
CHUNK = 20_000  # pixels per accumulator update, to bound memory


def weighted_median(values: FloatArray, weights: FloatArray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def iterative_hough(response: CentrelineResponse, road_mask: ArrayLike) -> list[FloatArray]:
    """Linear centreline segments, each a 2 x 2 array of end points (x, y)."""
    ys, xs = np.nonzero(np.asarray(road_mask, dtype=bool) & (response.magnitude > 0))
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    votes = response.magnitude[ys.astype(int), xs.astype(int)]
    widths = response.width[ys.astype(int), xs.astype(int)]

    phi = np.arange(N_ANGLES) * np.pi / N_ANGLES  # angle of the line normal
    cos, sin = np.cos(phi), np.sin(phi)
    diag = int(np.ceil(np.hypot(*response.magnitude.shape)))
    n_rho = 2 * diag + 1
    acc = np.zeros(N_ANGLES * n_rho)

    def add(idx: np.ndarray, sign: float) -> None:
        for chunk in np.array_split(idx, max(1, len(idx) // CHUNK)):
            rho = np.round(xs[chunk, None] * cos + ys[chunk, None] * sin).astype(int) + diag
            bins = (np.arange(N_ANGLES) * n_rho + rho).ravel()
            acc[:] += sign * np.bincount(bins, np.repeat(votes[chunk], N_ANGLES), minlength=acc.size)

    alive = np.ones(len(xs), dtype=bool)
    add(np.arange(len(xs)), 1.0)
    segments = []
    while alive.any():
        a, r = divmod(int(np.argmax(acc)), n_rho)
        rho = r - diag
        normal = np.array([cos[a], sin[a]])
        direction = np.array([-sin[a], cos[a]])
        dist = np.abs(xs * cos[a] + ys * sin[a] - rho)
        on_line = alive & (dist <= 1)
        if not on_line.any():  # only round-off left in the accumulator
            break
        road_width = weighted_median(widths[on_line], votes[on_line])
        near = np.flatnonzero(alive & (dist <= road_width / 2 + 1))
        t = np.sort(xs[near] * direction[0] + ys[near] * direction[1])
        breaks = np.flatnonzero(np.diff(t) > road_width)
        for run in np.split(t, breaks + 1):
            if run[-1] - run[0] >= road_width:
                segments.append(rho * normal + np.outer([run[0], run[-1]], direction))
        alive[near] = False
        add(near, -1.0)
    return segments


# ----------------------------------------------------------------------
# Sec. 6.2: road tracking (Eq. 21) and road segments
# ----------------------------------------------------------------------

def densify(vertices: FloatArray, step: float = 1.0) -> FloatArray:
    """Points every `step` pixels along a polyline, vertices included."""
    out = [vertices[:1]]
    for p, q in zip(vertices[:-1], vertices[1:]):
        n = max(1, int(np.ceil(np.hypot(*(q - p)) / step)))
        out.append(p + np.outer(np.arange(1, n + 1) / n, q - p))
    return np.concatenate(out)


def directions(points: FloatArray) -> FloatArray:
    """Unit direction of travel at each point of a dense polyline."""
    d = np.gradient(points, axis=0)
    norm = np.hypot(d[:, 0], d[:, 1])
    return d / np.where(norm > 0, norm, 1.0)[:, None]


def sample_info(points: FloatArray, response: CentrelineResponse) -> tuple[FloatArray, FloatArray]:
    """Width and orientation recorded at each point.

    Where the filters gave no response (a gap the Hough line bridges), the
    nearest point with a response along the centreline is used.
    """
    h, w = response.magnitude.shape
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    valid = np.flatnonzero(response.magnitude[rows, cols] > 0)
    if len(valid) == 0:
        return np.zeros(len(points)), np.zeros(len(points))
    nearest = valid[np.abs(np.arange(len(points))[:, None] - valid[None, :]).argmin(axis=1)]
    return response.width[rows[nearest], cols[nearest]], response.orientation[rows[nearest], cols[nearest]]


def road_sides(points: FloatArray, widths: FloatArray, thetas: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Right and left road-side points, Eq. (21).

    Orientations are undirected; theta + pi is used where it is nearer the
    direction of travel, so the right side stays on one side of the road.
    """
    d = directions(points)
    flip = np.cos(thetas) * d[:, 0] + np.sin(thetas) * d[:, 1] < 0
    theta = thetas + np.pi * flip
    offset = np.stack([widths * np.sin(theta) / 2, -widths * np.cos(theta) / 2], axis=1)
    return points + offset, points - offset


def road_segment(vertices: FloatArray, response: CentrelineResponse) -> BaseGeometry:
    """Polygon of one road segment, between its left and right road sides.

    Built as the union of one quadrilateral per step along the centreline,
    so a jump in the recorded orientation cannot turn the outline inside out.
    """
    points = densify(vertices)
    right, left = road_sides(points, *sample_info(points, response))
    quads = np.stack([left[:-1], left[1:], right[1:], right[:-1]], axis=1)
    return shapely.unary_union(shapely.make_valid(shapely.polygons(quads)))


def longest_network(centrelines: Sequence[FloatArray], segments: Sequence[BaseGeometry],
                    widths: Sequence[float]) -> list[int]:
    """Indices of the connected group of road segments with the greatest centreline length.

    Automatic mode keeps only the longest detected road network (Sec. 7).
    Two segments are connected when the gap between them is no longer than
    half the road width of the first (the same rule that splits a Hough line
    into segments, applied from both sides).
    """
    if not segments:
        return []
    parent = list(range(len(segments)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree(list(segments))
    grown = [s.buffer(w / 2) for s, w in zip(segments, widths)]
    for i, j in zip(*tree.query(grown, predicate="intersects")):
        parent[root(int(i))] = root(int(j))
    length: dict[int, float] = {}
    for i, c in enumerate(centrelines):
        length[root(i)] = length.get(root(i), 0.0) + LineString(c).length
    if not length:
        return []
    best = max(length, key=lambda k: length[k])
    return [i for i in range(len(segments)) if root(i) == best]


def segment_width(vertices: FloatArray, response: CentrelineResponse) -> float:
    """Median recorded width along a centreline."""
    return float(np.median(sample_info(densify(vertices), response)[0]))


# ----------------------------------------------------------------------
# Sec. 6.2: refinement actions 1-7 (Eqs. 22-27)
# ----------------------------------------------------------------------

# Eq. (22) weights, as given in the paper
W_THETA, W_D, W_W = 0.4, 0.3, 0.3
# Eq. (27) weights, as given in the paper (a comes from the curve map)
SNAKE_B, SNAKE_C = 0.5, 0.7
# Snake step size and iterations: not in the paper (PLAN.md, Assumptions)
SNAKE_GAMMA, SNAKE_ITERATIONS = 1.0, 200


@dataclass
class Inputs:
    """Road sides, Stage 2 mask and filter settings, needed to recompute centreline information."""
    saliency: FloatArray
    orientation: FloatArray
    road_mask: FloatArray
    widths: Sequence[float]


def angle_diff(a: FloatArray, b: FloatArray) -> FloatArray:
    """Smallest difference between undirected orientations, in [0, pi/2]."""
    d = np.mod(a - b, np.pi)
    return np.minimum(d, np.pi - d)


def recompute(vertices: FloatArray, response: CentrelineResponse, inputs: Inputs) -> CentrelineResponse:
    """Write the filters' answer along a centreline into the maps, turned to its local slope."""
    points = densify(vertices)
    d = directions(points)
    along = fixed_orientation_response(inputs.saliency, inputs.orientation, inputs.road_mask, points,
                                       np.arctan2(d[:, 1], d[:, 0]), inputs.widths)
    h, w = response.magnitude.shape
    cols = np.round(points[:, 0]).astype(int)
    rows = np.round(points[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    out = replace(response, magnitude=response.magnitude.copy(), orientation=response.orientation.copy(),
                  width=response.width.copy())
    r, c = rows[inside], cols[inside]
    out.magnitude[r, c] = along.magnitude[inside]
    out.orientation[r, c] = along.orientation[inside]
    out.width[r, c] = along.width[inside]
    return out


def track_from_seed(seed: Sequence[float], response: CentrelineResponse, road_mask: FloatArray) -> FloatArray:
    """Action 1: follow a road from a seed point both ways, Eq. (22).

    Each step predicts the next point one pixel along the current
    orientation; the candidates are the 3 x 3 road-mask pixels around that
    prediction with a centreline response, not yet on the line. The candidate that
    minimises w_d D + w_theta O + w_w W is added (D: distance from the
    prediction / sqrt 2, O: orientation difference / (pi/2), W: width
    difference / current width). Tracking stops when no candidate is left,
    or when the line comes back within half the road width of its own earlier
    points (more than one road width back), as around a roundabout
    (PLAN.md, Assumptions).
    """
    h, w = response.magnitude.shape
    start = np.round(np.asarray(seed, dtype=np.float64)).astype(int)
    visited = {(int(start[0]), int(start[1]))}
    halves: list[list[FloatArray]] = []
    for sign in (1.0, -1.0):
        p = start.astype(np.float64)
        theta = response.orientation[start[1], start[0]]
        d = sign * np.array([np.cos(theta), np.sin(theta)])
        chain: list[FloatArray] = []
        while True:
            pred = p + d
            cx, cy = np.meshgrid(np.arange(-1, 2) + round(pred[0]), np.arange(-1, 2) + round(pred[1]))
            cands = [(x, y) for x, y in zip(cx.ravel(), cy.ravel())
                     if 0 <= x < w and 0 <= y < h and (x, y) not in visited
                     and response.magnitude[y, x] > 0 and road_mask[y, x] > 0]
            if not cands:
                break
            cur = int(round(p[1])), int(round(p[0]))
            xy = np.array(cands, dtype=np.float64)
            rows, cols = xy[:, 1].astype(int), xy[:, 0].astype(int)
            cost = (W_D * np.hypot(*(xy - pred).T) / np.sqrt(2)
                    + W_THETA * angle_diff(response.orientation[rows, cols], response.orientation[cur]) / (np.pi / 2)
                    + W_W * np.abs(response.width[rows, cols] - response.width[cur]) / max(response.width[cur], 1.0))
            p = xy[int(np.argmin(cost))]
            half_width = max(response.width[int(p[1]), int(p[0])], 2.0) / 2
            earlier = [start.astype(np.float64)] + (halves[0] if halves else []) + chain[:-int(2 * half_width)]
            if np.hypot(*(np.array(earlier) - p).T).min() < half_width and len(chain) > 2 * half_width:
                break
            visited.add((int(p[0]), int(p[1])))
            chain.append(p)
            theta = response.orientation[int(p[1]), int(p[0])]
            new_d = np.array([np.cos(theta), np.sin(theta)])
            d = new_d if new_d @ d >= 0 else -new_d
        halves.append(chain)
    return np.array(halves[1][::-1] + [start.astype(np.float64)] + halves[0])


def hermite_merge(a: FloatArray, b: FloatArray, inputs: Inputs, response: CentrelineResponse) -> FloatArray:
    """Action 3: join two centrelines with a cubic Hermite curve, Eqs. (23)-(26).

    The end points joined are the closest pair. Their tangent directions are
    the Stage 1 (tensor voting) orientations there, or, where Stage 1 found no
    curve, the recorded centreline orientation. Tangent length is the
    distance between the end points (PLAN.md, Assumptions).
    """
    ends = [(a[::-1], b), (a[::-1], b[::-1]), (a, b), (a, b[::-1])]  # first ends where the gap starts
    first, second = min(ends, key=lambda e: float(np.hypot(*(e[0][-1] - e[1][0]))))
    p0, p1 = first[-1], second[0]
    chord = float(np.hypot(*(p1 - p0)))

    def tangent(p: FloatArray, way: FloatArray) -> FloatArray:
        r, c = int(round(p[1])), int(round(p[0]))
        r = min(max(r, 0), inputs.saliency.shape[0] - 1)
        c = min(max(c, 0), inputs.saliency.shape[1] - 1)
        theta = inputs.orientation[r, c] if inputs.saliency[r, c] > 0 else response.orientation[r, c]
        t = np.array([np.cos(theta), np.sin(theta)])
        return chord * (t if t @ way >= 0 else -t)

    gap = (p1 - p0) / chord if chord > 0 else np.zeros(2)
    o0 = tangent(p0, first[-1] - first[-2] if len(first) > 1 else gap)
    o1 = tangent(p1, second[1] - second[0] if len(second) > 1 else gap)
    m = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float64)  # Eq. (25)
    u = np.linspace(0, 1, max(2, int(np.ceil(chord)) + 1))
    curve = np.stack([u ** 3, u ** 2, u, np.ones_like(u)], axis=1) @ m @ np.stack([p0, p1, o0, o1])  # Eqs. (23)-(24)
    return np.concatenate([first, curve[1:-1], second])


def snake(vertices: FloatArray, inputs: Inputs, response: CentrelineResponse) -> FloatArray:
    """Action 5: smooth a centreline with a snake, Eq. (27).

    Elasticity a at each point is the Stage 1 curve saliency there; stiffness
    b = 0.5; the image term (weight c = 0.7) pulls the points up the
    normalised centreline magnitude map. Semi-implicit Kass iterations with
    the two end points fixed (PLAN.md, Assumptions).
    """
    v = densify(vertices)
    n = len(v)
    if n < 3:
        return v
    h, w = response.magnitude.shape
    rows = np.clip(np.round(v[:, 1]).astype(int), 0, h - 1)
    cols = np.clip(np.round(v[:, 0]).astype(int), 0, w - 1)
    a = inputs.saliency[rows, cols]
    d1 = np.diff(np.eye(n), axis=0)          # first differences, (n-1) x n
    d2 = np.diff(np.eye(n), n=2, axis=0)     # second differences, (n-2) x n
    stiffness = d1.T @ np.diag((a[:-1] + a[1:]) / 2) @ d1 + SNAKE_B * d2.T @ d2
    stiffness[[0, -1]] = 0  # fixed end points
    step = np.linalg.inv(stiffness + SNAKE_GAMMA * np.eye(n))
    m = response.magnitude / response.magnitude.max() if response.magnitude.max() > 0 else response.magnitude
    gy, gx = (np.asarray(g, dtype=np.float32) for g in np.gradient(m))
    for _ in range(SNAKE_ITERATIONS):
        px, py = v[:, 0].astype(np.float32)[None, :], v[:, 1].astype(np.float32)[None, :]
        force = np.stack([cv2.remap(gx, px, py, cv2.INTER_LINEAR)[0],
                          cv2.remap(gy, px, py, cv2.INTER_LINEAR)[0]], axis=1)
        force[[0, -1]] = 0
        v = step @ (SNAKE_GAMMA * v + SNAKE_C * force)
    return v


def reduce_points(vertices: FloatArray, max_error: float) -> FloatArray:
    """Action 7: Iterative End-Point Fit (Douglas-Peucker) with the user's maximum error."""
    approx = cv2.approxPolyDP(vertices.astype(np.float32).reshape(-1, 1, 2), max_error, closed=False)
    return np.asarray(approx.reshape(-1, 2), dtype=np.float64)


def refine(centrelines: dict[int, FloatArray], response: CentrelineResponse, inputs: Inputs,
           actions: Sequence[dict[str, Any]]) -> tuple[dict[int, FloatArray], CentrelineResponse]:
    """Apply the actions in order. Returns the refined centrelines and centreline information.

    Actions come from an edit file instead of mouse clicks:

    {"actions": [
        {"action": "seed", "point": [x, y]},
        {"action": "add", "points": [[x, y], ...]},
        {"action": "edit", "id": 3, "points": [[x, y], ...]},
        {"action": "merge", "ids": [3, 7]},
        {"action": "delete", "id": 5},
        {"action": "smooth", "id": 3},
        {"action": "reduce", "id": 3, "max_error": 2.0}
    ]}

    Ids are those of the automatic centrelines; a seed or
    an added centreline gets the next free id. After every change except a
    delete, the centreline information (magnitude, orientation, width) is
    recomputed along the changed centreline with the filters turned to its
    local slope (action 2), as in Fig. 15. A delete leaves it unchanged.
    """
    lines = dict(centrelines)
    next_id = max(lines, default=-1) + 1
    for act in actions:
        kind = act["action"]
        if kind == "delete":
            del lines[act["id"]]
            continue
        if kind == "seed":
            key, next_id = next_id, next_id + 1
            lines[key] = track_from_seed(act["point"], response, inputs.road_mask)
        elif kind == "add":
            key, next_id = next_id, next_id + 1
            lines[key] = np.asarray(act["points"], dtype=np.float64)
        elif kind == "edit":
            key = act["id"]
            lines[key] = np.asarray(act["points"], dtype=np.float64)
        elif kind == "merge":
            key, other = act["ids"]
            lines[key] = hermite_merge(lines[key], lines.pop(other), inputs, response)
        elif kind == "smooth":
            key = act["id"]
            lines[key] = snake(lines[key], inputs, response)
        elif kind == "reduce":
            key = act["id"]
            lines[key] = reduce_points(lines[key], act["max_error"])
        else:
            raise ValueError(f"unknown action {kind!r}")
        response = recompute(lines[key], response, inputs)
    return lines, response


# ----------------------------------------------------------------------
# Sec. 6: automatic network
# ----------------------------------------------------------------------

def automatic(response: CentrelineResponse, road_mask: ArrayLike) -> dict[int, FloatArray]:
    """Linear centreline segments of the longest connected road network, numbered from 0."""
    lines = iterative_hough(response, road_mask)
    segments = [road_segment(c, response) for c in lines]
    keep = longest_network(lines, segments, [segment_width(c, response) for c in lines])
    return {i: lines[k] for i, k in enumerate(keep)}


def track(centrelines: dict[int, FloatArray], response: CentrelineResponse) -> BaseGeometry:
    """Road segments from Eq. (21), merged by a polygon union so junctions overlap cleanly."""
    return shapely.unary_union([road_segment(c, response) for c in centrelines.values()])




# ----------------------------------------------------------------------
# Mask cleaning and centreline post-processing
# ----------------------------------------------------------------------

def line_kernel(length: int, theta: float) -> NDArray[np.uint8]:
    kernel = np.zeros((length, length), np.uint8)
    c = length // 2
    dx, dy = np.cos(theta) * c, np.sin(theta) * c
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return kernel


def line_opening(mask: ArrayLike, length: float, n_orientations: int) -> BoolArray:
    """Union of openings with straight lines: keeps structures at least `length` long in some direction."""
    mask = np.asarray(mask).astype(np.uint8)
    length = int(length) | 1
    result = np.zeros_like(mask)
    for theta in np.arange(n_orientations) * np.pi / n_orientations:
        result |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, line_kernel(length, theta))
    return result.astype(bool)


def clean_mask(mask: ArrayLike, min_width: float, max_width: float, n_orientations: int = 24) -> BoolArray:
    """Keep long, wide-enough mask structures before every mask pixel votes in the Hough transform.

    The union of openings with straight lines about 1.4 x max_width long, over
    n_orientations, keeps roads and drops blobs on roofs and in courtyards
    (a path-opening road filter). A disc of diameter 0.7 x min_width then drops
    thin strips, such as building edges, that survive the line openings.
    """
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(np.asarray(mask).astype(np.uint8), cv2.MORPH_CLOSE, small)  # bridge cars and lane marks
    linear = line_opening(mask, round(1.4 * max_width), n_orientations).astype(np.uint8)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(0.7 * min_width) | 1,) * 2)
    return cv2.morphologyEx(linear, cv2.MORPH_OPEN, disc).astype(bool)


def mask_width(vertices: ArrayLike, distance: FloatArray) -> float:
    """Road width along a centreline: twice the median distance to the mask edge.

    Unlike the filter's winning width, this does not saturate at the largest
    tested width. Points off the mask (gaps the line bridges) are ignored.
    """
    h, w = distance.shape
    points = np.rint(densify(np.asarray(vertices, float))).astype(int)
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    values = distance[points[inside, 1], points[inside, 0]]
    values = values[values > 0]
    return 2 * float(np.median(values)) if values.size else 0.0


def split_off_mask(vertices: ArrayLike, mask: BoolArray, max_gap: float, min_length: float) -> list[FloatArray]:
    """Cut a straight segment where it leaves the road mask for more than max_gap pixels.

    The Hough transform splits lines only at gaps wider than the filter's road
    width, which saturates at the largest tested width, so its segments can
    cross whole blocks. Pieces shorter than min_length are dropped.
    """
    h, w = mask.shape
    points = densify(np.asarray(vertices, float))
    cols = np.clip(np.rint(points[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(int), 0, h - 1)
    on = np.flatnonzero(mask[rows, cols])
    if on.size == 0:
        return []
    runs = np.split(on, np.flatnonzero(np.diff(on) > max_gap) + 1)
    return [points[[run[0], run[-1]]] for run in runs if run[-1] - run[0] >= min_length]


def merge_parallel(lines, width_of, max_angle=np.radians(6)):
    """Merge Hough segments that lie side by side on one road.

    On a wide road the iterative Hough transform returns several near-parallel
    segments, one per lane band. Two segments merge when their angles differ by
    less than max_angle, their bands overlap across the road and their extents
    overlap or nearly meet along it. The merged segment lies midway across both
    bands and spans both extents. width_of(vertices) measures each segment's
    width, so a merged width comes from the image rather than growing with
    every merge.
    """
    lines = [np.asarray(line, float) for line in lines]
    widths = [width_of(line) for line in lines]
    merged = True
    while merged:
        merged = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                joined = _join(lines[i], widths[i], lines[j], widths[j], max_angle)
                if joined is not None:
                    lines[i], widths[i] = joined, width_of(joined)
                    del lines[j], widths[j]
                    merged = True
                    break
            if merged:
                break
    return lines, widths


def _join(a, width_a, b, width_b, max_angle):
    da, db = a[1] - a[0], b[1] - b[0]
    length_a, length_b = np.hypot(*da), np.hypot(*db)
    if length_a == 0 or length_b == 0:
        return None
    da, db = da / length_a, db / length_b
    if abs(da @ db) < np.cos(max_angle):
        return None
    if da @ db < 0:
        db = -db
    direction = da * length_a + db * length_b
    direction /= np.hypot(*direction)
    normal = np.array([-direction[1], direction[0]])
    origin = a[0]
    along_a, along_b = (a - origin) @ direction, (b - origin) @ direction
    across_a, across_b = (a - origin) @ normal, (b - origin) @ normal
    gap = max(along_a.min(), along_b.min()) - min(along_a.max(), along_b.max())
    if gap > max(width_a, width_b):
        return None
    low = min(across_a.min() - width_a / 2, across_b.min() - width_b / 2)
    high = max(across_a.max() + width_a / 2, across_b.max() + width_b / 2)
    if high - low >= width_a + width_b:
        return None  # bands do not overlap: separate roads
    centre = (low + high) / 2
    ends = [min(along_a.min(), along_b.min()), max(along_a.max(), along_b.max())]
    return origin + np.outer(ends, direction) + centre * normal


def extend_to_junctions(lines, widths, reach=2.0):
    """Extend each segment end to the nearest other centreline ahead of it.

    Hough segments stop short of junctions, where the road mask is widest and
    least straight. An end extends along its own direction by at most
    reach x its width; if no other centreline lies within that, it stays put.
    """
    lines = [np.asarray(line, float).copy() for line in lines]
    shapes = [LineString(line) for line in lines]
    for i, line in enumerate(lines):
        others = shapely.union_all([shape for j, shape in enumerate(shapes) if j != i])
        for end, inner in ((0, 1), (-1, -2)):
            direction = line[end] - line[inner]
            length = np.hypot(*direction)
            if length == 0 or others.is_empty:
                continue
            start = line[end]
            ray = LineString([start, start + direction / length * reach * max(widths[i], 1.0)])
            hit = ray.intersection(others)
            if not hit.is_empty:
                line[end] = np.asarray(shapely.ops.nearest_points(shapely.Point(start), hit)[1].coords[0])
    return lines


def polygon_parts(geometry: BaseGeometry):
    """Yield polygon parts, including those nested in geometry collections."""
    if geometry.geom_type == "Polygon":
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def polygon_mask(geometry: BaseGeometry, shape: tuple[int, int]) -> BoolArray:
    result = np.zeros(shape, np.uint8)
    for polygon in polygon_parts(geometry):
        part = np.zeros(shape, np.uint8)
        cv2.fillPoly(part, [np.rint(polygon.exterior.coords).astype(np.int32)], 1)
        for ring in polygon.interiors:
            cv2.fillPoly(part, [np.rint(ring.coords).astype(np.int32)], 0)
        result |= part
    return result.astype(bool)


# ----------------------------------------------------------------------
# Pipeline stage
# ----------------------------------------------------------------------

def extract(image, curve_map, theta, road, widths=range(10, 61, 4), n_orientations=12,
            sigma_across=1.5, sigma_along=6.0, curve_source="tensor", keep_longest=True,
            clean=True, actions=None, progress=None):
    """Automatic road network from the Stage 1 curve map and the Stage 2 road mask.

    image         BGR uint8 input, used for the overlay.
    curve_map     Stage 1 curve saliency, 0 off curves; theta its tangent in [0, pi).
    road          Stage 2 boolean road mask.
    widths        road widths to test, in pixels.
    curve_source  "tensor" uses curve_map/theta as road sides; "mask" uses
                  the mask edges instead, as the standalone Stage 3 script did.
    keep_longest  keep only the longest connected network (Sec. 7).
    clean         clean the road mask with line openings first.
    actions       optional refinement actions, see `refine`; ids are those
                  of the automatic centrelines.
    """
    widths = [float(w) for w in widths]
    if not widths or min(widths) <= 0:
        raise ValueError("Use at least one positive road width.")
    if int(n_orientations) != n_orientations or n_orientations < 1:
        raise ValueError("Use a positive whole number of orientations.")
    if not (np.isfinite(sigma_across) and np.isfinite(sigma_along) and sigma_across > 0 and sigma_along > 0):
        raise ValueError("Use positive, finite filter sigmas.")
    if curve_source not in ("tensor", "mask"):
        raise ValueError('Use curve_source "tensor" or "mask".')
    road = np.asarray(road, dtype=bool)
    if road.shape != np.shape(curve_map) or road.shape != np.shape(theta) or road.shape != image.shape[:2]:
        raise ValueError("Image, curve map, orientations and road mask must have the same size.")
    report = progress or (lambda message: None)
    mask = clean_mask(road, min(widths), max(widths)) if clean else road
    if curve_source == "mask":
        saliency, orientation = curves_from_mask(mask)
    else:
        saliency, orientation = np.asarray(curve_map, np.float64), np.asarray(theta, np.float64)
    response = centreline_response(saliency, orientation, mask, widths, n_orientations=int(n_orientations),
                                   sigma_across=sigma_across, sigma_along=sigma_along, progress=report)
    report("iterative Hough transform")
    found = automatic(response, mask) if keep_longest else dict(enumerate(iterative_hough(response, mask)))
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    def width_of(vertices):
        return mask_width(vertices, distance) or segment_width(vertices, response)

    pieces = [piece for c in found.values() for piece in split_off_mask(c, mask, min(widths), 2 * min(widths))]
    merged, line_widths = merge_parallel(pieces, width_of)
    lines = dict(enumerate(extend_to_junctions(merged, line_widths)))
    if actions:
        report("refinement actions")
        inputs = Inputs(saliency, orientation, mask.astype(np.float64), widths)
        lines, response = refine(lines, response, inputs, actions)
        line_widths = [width_of(c) for c in lines.values()]
    report("road tracking and polygon union")
    # Polygonal output is the footprint; degenerate lines can occur in a union.
    network = shapely.unary_union(list(polygon_parts(track(lines, response))))
    centrelines = [{"id": key, "points": np.asarray(c).tolist(), "width": float(w)}
                   for (key, c), w in zip(lines.items(), line_widths)]
    centreline_mask = np.zeros(road.shape, np.uint8)
    overlay = image.copy()
    for polygon in polygon_parts(network):
        for ring in (polygon.exterior, *polygon.interiors):
            cv2.polylines(overlay, [np.rint(ring.coords).astype(np.int32)], True, (0, 255, 0), 1)
    for c in lines.values():
        points = np.rint(c).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(centreline_mask, [points], False, 1, 1)
        cv2.polylines(overlay, [points], False, (0, 255, 255), 1)
    warnings = []
    if not lines:
        warnings.append("No centreline segments were found. Check the segmentation, curve source and widths.")
    if max(widths) >= min(road.shape):
        warnings.append("Maximum road width reaches the shorter image side; consider smaller widths.")
    return {
        "cleaned_road": mask,
        "curve_saliency": saliency,
        "magnitude": response.magnitude,
        "orientation": response.orientation,
        "width": response.width,
        "centreline_mask": centreline_mask.astype(bool),
        "network_mask": polygon_mask(network, road.shape),
        "overlay": overlay,
        "centrelines": centrelines,
        "road_network": mapping(network),
        "centreline_count": len(lines),
        "total_length_px": float(sum(LineString(c).length for c in lines.values())),
        "polygon_area_px2": float(network.area),
        "widths": widths,
        "warnings": warnings,
    }
