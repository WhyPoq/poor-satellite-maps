import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from centerline import WIDTHS, kernels, make_segment, orientation_weight, segment_length

W_THETA, W_DIST, W_WIDTH = (0.4, 0.3, 0.3)


def tangents(pts):
    """Per-vertex tangent orientation (mod pi) of a polyline."""
    d = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[1.0, 0.0]])
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)


def from_points(pts, width):
    pts = np.asarray(pts, float)
    return make_segment(
        pts, tangents(pts), np.broadcast_to(np.asarray(width, float), len(pts)).copy()
    )


def sample(img, pts, order=1):
    return map_coordinates(img, [pts[:, 1], pts[:, 0]], order=order, mode="nearest")


def filter_at(ctx, pts, phi, widths=WIDTHS, **kernel_parameters):
    """Apply Gt = Gb * Gs at points for a fixed orientation phi; return (best width, response) per point."""
    C = ctx["curve_points"] * orientation_weight(ctx["theta"], phi)
    road = ctx["road"].astype(np.float32)
    h, w = road.shape
    best_w = np.full(len(pts), float(widths[0]))
    best_r = np.zeros(len(pts))
    for wd in widths:
        gb, gs = kernels(phi, wd, **kernel_parameters)
        r = gb.shape[0] // 2
        for i, (x, y) in enumerate(np.round(pts).astype(int)):
            y0, y1, x0, x1 = (y - r, y + r + 1, x - r, x + r + 1)
            if y0 < 0 or x0 < 0 or y1 > h or (x1 > w):
                continue
            resp = (gb * C[y0:y1, x0:x1]).sum() * (gs * road[y0:y1, x0:x1]).sum()
            if resp > best_r[i]:
                best_r[i], best_w[i] = (resp, wd)
    return (best_w, best_r)


def add_seed(
    ctx,
    seed,
    max_steps=2000,
    n_orient=8,
    widths=WIDTHS,
    w_dist=W_DIST,
    w_theta=W_THETA,
    w_width=W_WIDTH,
    **kernel_parameters,
):
    """(1) Track a road from a seed point by recursive neighbourhood search minimising Eq 22."""
    M, Phi, Wm, road = (ctx["M"], ctx["Phi"], ctx["W"], ctx["road"])
    h, w = M.shape
    x, y = np.round(seed).astype(int)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Seed is outside the image.")
    theta0, width0 = (float(Phi[y, x]), float(Wm[y, x]))
    if M[y, x] <= 0:
        cands = [
            filter_at(ctx, np.array([[x, y]], float), p, widths, **kernel_parameters)
            + (p,)
            for p in np.arange(n_orient) * np.pi / n_orient
        ]
        bw, _br, theta0 = max(cands, key=lambda c: c[1][0])
        width0 = float(bw[0])
    halves = []
    for sign in (1, -1):
        p, th, wd = (np.array([x, y], float), theta0, width0)
        direction = sign * np.array([np.cos(th), np.sin(th)])
        path, visited = ([], {(x, y)})
        for _ in range(max_steps):
            step = max(2.0, wd / 2)
            r = int(np.ceil(step))
            yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
            cx, cy = np.round(p).astype(int)[:, None, None] + np.stack([xx, yy])
            ok = (cx >= 0) & (cy >= 0) & (cx < w) & (cy < h)
            cx, cy = (cx[ok], cy[ok])
            d = np.hypot(cx - p[0], cy - p[1])
            ahead = (cx - p[0]) * direction[0] + (cy - p[1]) * direction[1] > 0.5 * d
            good = (
                ahead & (d > 0.5 * step) & (d <= step) & (M[cy, cx] > 0) & road[cy, cx]
            )
            good &= np.array([(a, b) not in visited for a, b in zip(cx, cy)], bool)
            if not good.any():
                break
            cx, cy, d = (cx[good], cy[good], d[good])
            dtheta = np.abs(Phi[cy, cx] - th) % np.pi
            o = np.minimum(dtheta, np.pi - dtheta) / (np.pi / 2)
            cost = (
                w_dist * d / step
                + w_theta * o
                + w_width * np.minimum(np.abs(Wm[cy, cx] - wd) / max(wd, 1e-12), 1)
            )
            i = int(np.argmin(cost))
            q = np.array([cx[i], cy[i]], float)
            visited.add((cx[i], cy[i]))
            direction = (q - p) / np.linalg.norm(q - p)
            p, th, wd = (q, float(Phi[cy[i], cx[i]]), float(Wm[cy[i], cx[i]]))
            path.append((p.copy(), wd))
        halves.append(path)
    pts = (
        [q for q, _ in halves[1][::-1]]
        + [np.array([x, y], float)]
        + [q for q, _ in halves[0]]
    )
    wds = [v for _, v in halves[1][::-1]] + [width0] + [v for _, v in halves[0]]
    seg = from_points(np.array(pts), np.array(wds))
    return seg


def add_centerline(ctx, p0, p1, spacing=5.0, widths=WIDTHS, **kernel_parameters):
    """(2) User-drawn centerline: filters at the fixed orientation of the line give per-point widths."""
    p0, p1 = (np.asarray(p0, float), np.asarray(p1, float))
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / spacing)) + 1)
    pts = np.linspace(p0, p1, n)
    phi = np.mod(np.arctan2(*(p1 - p0)[::-1]), np.pi)
    best_widths, _ = filter_at(ctx, pts, phi, widths, **kernel_parameters)
    return make_segment(pts, np.full(n, phi), best_widths)


def delete_centerline(segments, index):
    """(4) Remove a segment. The underlying M / Phi / W maps are untouched."""
    return [s for i, s in enumerate(segments) if i != index]


def hermite(p0, p1, t0, t1, n=20):
    """Eq 23-26: P(u) = U^T M B with B = [P_i, P_i+1, O_i, O_i+1]."""
    Mh = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]], float)
    u = np.linspace(0, 1, n)
    U = np.stack([u**3, u**2, u, np.ones_like(u)], 1)
    B = np.stack([p0, p1, t0, t1])
    return U @ Mh @ B


def merge(ctx, a, b, n=20):
    """(3) Join two segments with a Hermite curve between their closest endpoints."""
    ends = [(i, j) for i in (0, -1) for j in (0, -1)]
    i, j = min(ends, key=lambda e: np.linalg.norm(a["pts"][e[0]] - b["pts"][e[1]]))
    pa, pb = (a["pts"][i], b["pts"][j])
    chord = pb - pa
    gap = np.linalg.norm(chord)

    def tangent_at(p):
        th = float(sample(ctx["theta"], p[None])[0])
        t = np.array([np.cos(th), np.sin(th)])
        return gap * (t if t @ chord >= 0 else -t)

    curve = hermite(pa, pb, tangent_at(pa), tangent_at(pb), n)[1:-1]
    a_pts = a["pts"] if i == -1 else a["pts"][::-1]
    a_w = a["width"] if i == -1 else a["width"][::-1]
    b_pts = b["pts"] if j == 0 else b["pts"][::-1]
    b_w = b["width"] if j == 0 else b["width"][::-1]
    w_mid = np.linspace(a_w[-1], b_w[0], len(curve) + 2)[1:-1]
    return from_points(
        np.vstack([a_pts, curve, b_pts]), np.concatenate([a_w, w_mid, b_w])
    )


def densify(seg, spacing=1.0):
    if spacing <= 0:
        raise ValueError("Spacing must be positive.")
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg["pts"], axis=0), axis=1))]
    s = np.linspace(0, d[-1], max(2, int(np.ceil(d[-1] / spacing)) + 1))
    pts = np.stack([np.interp(s, d, seg["pts"][:, k]) for k in range(2)], 1)
    return from_points(pts, np.interp(s, d, seg["width"]))


def smooth(ctx, seg, b=0.5, c=0.7, iters=200, tau=0.5, spacing=1.0, sigma=1.5):
    """(5) Snake on the centerline magnitude map (Eq 27), endpoints fixed."""
    seg = densify(seg, spacing)
    pts = seg["pts"].copy()
    n = len(pts)
    if n < 5:
        return seg
    sal = ctx["curve_sal"] / max(ctx["curve_sal"].max(), 1e-12)
    a = sample(sal, pts)
    D1 = diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    D2 = diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
    )
    a_edge = (a[:-1] + a[1:]) / 2
    A = (D1.T @ diags(a_edge) @ D1 + b * D2.T @ D2).tolil()
    A[0, :] = 0
    A[n - 1, :] = 0
    lu = splu((identity(n, format="csc") + tau * A.tocsc()).tocsc())
    Mn = gaussian_filter(ctx["M"] / max(ctx["M"].max(), 1e-12), sigma)
    gy, gx = np.gradient(Mn)
    gmax = max(np.hypot(gx, gy).max(), 1e-12)
    gx, gy = (gx / gmax, gy / gmax)
    for _ in range(iters):
        f = np.stack([sample(gx, pts), sample(gy, pts)], 1)
        f[[0, -1]] = 0
        pts = np.stack([lu.solve(pts[:, k] + tau * c * f[:, k]) for k in range(2)], 1)
    return from_points(pts, seg["width"])


def simplify(seg, tol=1.5):
    """(7) Iterative End-Point Fit (Douglas-Peucker) with maximum error `tol` px."""
    if len(seg["pts"]) < 2:
        return seg
    line = LineString(seg["pts"]).simplify(tol, preserve_topology=False)
    pts = np.asarray(line.coords)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg["pts"], axis=0), axis=1))]
    s = np.array([LineString(seg["pts"]).project(Point(p)) for p in pts])
    return make_segment(pts, tangents(pts), np.interp(s, d, seg["width"]))


def split(seg, at):
    """Split a segment at the vertex nearest to point `at`; returns two segments sharing that vertex."""
    if len(seg["pts"]) == 2:
        midpoint = (seg["pts"][0] + seg["pts"][1]) / 2
        width = float(np.mean(seg["width"]))
        return [
            from_points([seg["pts"][0], midpoint], [seg["width"][0], width]),
            from_points([midpoint, seg["pts"][1]], [width, seg["width"][1]]),
        ]
    if len(seg["pts"]) < 2:
        raise ValueError("A segment needs at least two points to split.")
    i = int(np.argmin(np.linalg.norm(seg["pts"] - np.asarray(at, float), axis=1)))
    i = min(max(i, 1), len(seg["pts"]) - 2) if len(seg["pts"]) > 2 else 1
    parts = [slice(0, i + 1), slice(i, None)]
    return [
        make_segment(seg["pts"][p], seg["theta"][p], seg["width"][p]) for p in parts
    ]


def connection_supported(road, start, end, min_support=0.7):
    """Require most of a proposed connection to lie on segmented road."""
    if road is None:
        return True
    count = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
    points = np.linspace(start, end, count)
    values = map_coordinates(
        road, [points[:, 1], points[:, 0]], order=0, mode="constant", cval=0
    )
    return values.mean() >= min_support


def join_collinear(
    segments, max_angle=np.pi / 12, max_gap=6.0, road=None, min_support=0.7
):
    """Join nearby segments whose endpoint directions agree."""
    segments = [s for s in segments if len(s["pts"]) >= 2 and segment_length(s) > 0]
    segs = [
        make_segment(
            s["pts"].copy(), s["theta"].copy(), s["width"].copy(), dict(s["meta"])
        )
        for s in segments
    ]
    while True:
        best = None
        for i, a in enumerate(segs):
            for j, b in enumerate(segs):
                if j <= i:
                    continue
                for ea in (0, -1):
                    for eb in (0, -1):
                        pa, pb = (a["pts"][ea], b["pts"][eb])
                        da = pa - a["pts"][1 if ea == 0 else -2]
                        db = pb - b["pts"][1 if eb == 0 else -2]
                        if min(np.linalg.norm(da), np.linalg.norm(db)) == 0:
                            continue
                        da, db = (da / np.linalg.norm(da), db / np.linalg.norm(db))
                        gap_v = pb - pa
                        gap = np.linalg.norm(gap_v)
                        w = (a["width"][ea] + b["width"][eb]) / 2
                        if (
                            gap > max_gap * w
                            or np.arccos(np.clip(-da @ db, -1, 1)) > max_angle
                        ):
                            continue
                        lateral = abs(da[0] * gap_v[1] - da[1] * gap_v[0])
                        if lateral > w / 2 or gap_v @ da < -w:
                            continue
                        if not connection_supported(road, pa, pb, min_support):
                            continue
                        if best is None or gap < best[0]:
                            best = (gap, i, j, ea, eb)
        if best is None:
            return segs
        _, i, j, ea, eb = best
        a, b = (segs[i], segs[j])
        a_pts, a_w = (
            (a["pts"], a["width"]) if ea == -1 else (a["pts"][::-1], a["width"][::-1])
        )
        b_pts, b_w = (
            (b["pts"], b["width"]) if eb == 0 else (b["pts"][::-1], b["width"][::-1])
        )
        pts = np.vstack([a_pts, b_pts])
        segs[i] = make_segment(pts, tangents(pts), np.concatenate([a_w, b_w]))
        del segs[j]


def snap_and_prune(
    segments,
    reach=8.0,
    max_angle=np.pi / 12,
    max_gap=6.0,
    keep_longest=False,
    forward_cos=0.7,
    road=None,
    min_support=0.7,
):
    """Connect supported segments; optionally keep only the longest network."""
    segments = join_collinear(segments, max_angle, max_gap, road, min_support)
    if not segments:
        return []
    w_typ = float(np.median(np.concatenate([s["width"] for s in segments])))
    lines = [LineString(s["pts"]) for s in segments]
    g = nx.Graph()
    g.add_nodes_from(range(len(segments)))
    out = [
        make_segment(
            s["pts"].copy(), s["theta"].copy(), s["width"].copy(), dict(s["meta"])
        )
        for s in segments
    ]
    for i, s in enumerate(segments):
        for e, nb in ((0, 1), (-1, -2)):
            p, w = (s["pts"][e], float(s["width"][e]))
            d = p - s["pts"][nb]
            d /= max(np.linalg.norm(d), 1e-09)
            ray = LineString([p, p + reach * w_typ * d])
            best = None
            for j, ln in enumerate(lines):
                if j == i:
                    continue
                q = ln.interpolate(ln.project(Point(p)))
                to_q = np.array([q.x, q.y]) - p
                dist = np.linalg.norm(to_q)
                if dist <= w and (dist <= w / 4 or to_q @ d >= forward_cos * dist):
                    cand = (dist, j, q)
                else:
                    hit = ray.intersection(ln.buffer(w / 2))
                    if hit.is_empty:
                        continue
                    first = nearest_points(hit, Point(p))[0]
                    q = ln.interpolate(ln.project(first))
                    cand = (Point(p).distance(q), j, q)
                target = np.array([cand[2].x, cand[2].y])
                if not connection_supported(road, p, target, min_support):
                    continue
                if best is None or cand[0] < best[0]:
                    best = cand
            if best is not None:
                g.add_edge(i, best[1])
                q = np.array([[best[2].x, best[2].y]])
                if np.linalg.norm(q - p) > 0.5:
                    seg = out[i]
                    ext = (
                        (lambda a, b: np.vstack([b, a]))
                        if e == 0
                        else lambda a, b: np.vstack([a, b])
                    )
                    seg["pts"] = ext(seg["pts"], q)
                    seg["theta"] = tangents(seg["pts"])
                    seg["width"] = ext(seg["width"][:, None], seg["width"][[e], None])[
                        :, 0
                    ]
    if not keep_longest:
        return out
    comp = max(
        nx.connected_components(g),
        key=lambda c: sum(segment_length(out[k]) for k in c),
    )
    return [out[k] for k in sorted(comp)]
