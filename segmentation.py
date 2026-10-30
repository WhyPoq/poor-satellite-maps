import maxflow
import numpy as np
from sklearn.mixture import GaussianMixture


def validate_seeds(seeds, shape):
    """Return unique integer (x, y) points for both classes."""
    points = {}
    for name in ("bg", "fg"):
        values = np.asarray(seeds.get(name, []))
        if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
            raise ValueError(f"Seeds '{name}' must contain (x, y) points.")
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("Seed coordinates must be finite integers.")
        values = np.unique(values.astype(int), axis=0)
        if np.any(values < 0) or np.any(values >= np.array(shape[::-1])):
            raise ValueError("Seed coordinates are outside the image.")
        points[name] = values
    if set(map(tuple, points["bg"])) & set(map(tuple, points["fg"])):
        raise ValueError("Foreground and background seeds overlap.")
    return points


def fit_gmm(samples, max_k=8, seed=0, reg_covar=1.0):
    """Choose the number of Gaussian components using BIC."""
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 1:
        samples = np.repeat(samples, 2, axis=0)
    count = max(1, min(max_k, len(samples) // 10, len(np.unique(samples, axis=0))))
    models = [
        GaussianMixture(k, reg_covar=reg_covar, random_state=seed).fit(samples)
        for k in range(1, count + 1)
    ]
    return min(models, key=lambda model: model.bic(samples))


def orient_dist(a, b):
    """Circular orientation distance in [0, 1]."""
    distance = np.abs(a - b) % np.pi
    return np.minimum(distance, np.pi - distance) / (np.pi / 2)


def shift_pair(shape, dy, dx):
    def slices(offset, size):
        if offset >= 0:
            return slice(0, size - offset), slice(offset, size)
        return slice(-offset, size), slice(0, size + offset)

    py, qy = slices(dy, shape[0])
    px, qx = slices(dx, shape[1])
    return (py, px), (qy, qx)


def data_costs(
    features,
    theta_sparse,
    seeds,
    n_theta=8,
    max_k=8,
    seed=0,
    reg_covar=1.0,
    hard_cost=1e6,
):
    """Return unary costs and the foreground color probability."""
    shape = features.shape[:2]
    seeds = validate_seeds(seeds, shape)
    samples = features.reshape(-1, features.shape[-1]).astype(float)
    models = [
        fit_gmm(features[p[:, 1], p[:, 0]], max_k, seed, reg_covar)
        for p in (seeds["bg"], seeds["fg"])
    ]
    likelihood = np.stack([model.score_samples(samples) for model in models])
    log_posterior = likelihood - np.logaddexp(likelihood[0], likelihood[1])
    color_cost = (1 - log_posterior).reshape(2, *shape)
    angles = np.arange(n_theta) * np.pi / n_theta
    distance = orient_dist(np.nan_to_num(theta_sparse)[None], angles[:, None, None])
    distance[:, np.isnan(theta_sparse)] = 0
    costs = (color_cost[:, None] / (2 - distance**2)).reshape(2 * n_theta, *shape)
    for name, wrong, right in (
        ("fg", slice(0, n_theta), slice(n_theta, None)),
        ("bg", slice(n_theta, None), slice(0, n_theta)),
    ):
        x, y = seeds[name].T
        costs[wrong, y, x] = hard_cost
        costs[right, y, x] = 0
    return costs, np.exp(log_posterior[1]).reshape(shape)


def edge_weights(
    gray, theta, lam=0.25, neighbours=((0, 1), (1, 0), (1, 1), (1, -1)), sigma=None
):
    """Contrast weights for each undirected neighbour pair."""
    if sigma is not None and sigma <= 0:
        raise ValueError("Contrast sigma must be positive.")
    if not neighbours or any(
        (dy == dx == 0) or max(abs(dy), abs(dx)) > 1 for dy, dx in neighbours
    ):
        raise ValueError("Use nonzero neighbour offsets within a 3 by 3 window.")
    if len({(dy, dx) for dy, dx in neighbours}) != len(neighbours) or any(
        (-dy, -dx) in neighbours for dy, dx in neighbours
    ):
        raise ValueError("Specify each undirected neighbour pair only once.")
    differences = []
    for dy, dx in neighbours:
        p, q = shift_pair(gray.shape, dy, dx)
        delta = (1 + (gray[p] - gray[q]) ** 2) / (
            2 - orient_dist(theta[p], theta[q]) ** 2
        )
        differences.append(delta - 0.5)
    variance = (
        np.mean(np.concatenate([d.ravel() ** 2 for d in differences]))
        if sigma is None
        else sigma**2
    )
    variance = max(variance, 1e-12)
    return [
        lam / np.hypot(dy, dx) * np.exp(-(d**2) / (2 * variance))
        for (dy, dx), d in zip(neighbours, differences)
    ]


def label_cost(a, b, n_theta):
    class_a, angle_a = np.divmod(a, n_theta)
    class_b, angle_b = np.divmod(b, n_theta)
    distance = orient_dist(angle_a * np.pi / n_theta, angle_b * np.pi / n_theta)
    return (1 + (class_a - class_b) ** 2) / (2 - distance**2) - 0.5


def energy(labels, costs, weights, neighbours, n_theta):
    total = np.take_along_axis(costs, labels[None], 0).sum(dtype=float)
    for (dy, dx), weight in zip(neighbours, weights):
        p, q = shift_pair(labels.shape, dy, dx)
        total += (weight * label_cost(labels[p], labels[q], n_theta)).sum()
    return float(total)


def expand(labels, alpha, costs, weights, neighbours, n_theta):
    """One graph-cut move; truncate non-submodular pairwise costs."""
    switch = costs[alpha].copy()
    keep = np.take_along_axis(costs, labels[None], 0)[0].copy()
    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes(labels.shape)
    for (dy, dx), weight in zip(neighbours, weights):
        p, q = shift_pair(labels.shape, dy, dx)
        both_keep = weight * label_cost(labels[p], labels[q], n_theta)
        q_switch = weight * label_cost(labels[p], alpha, n_theta)
        p_switch = weight * label_cost(alpha, labels[q], n_theta)
        for region, linear in ((p, p_switch - both_keep), (q, -p_switch)):
            switch[region] += np.maximum(linear, 0)
            keep[region] += np.maximum(-linear, 0)
        edge = np.zeros(labels.shape)
        edge[p] = np.maximum(q_switch + p_switch - both_keep, 0)
        structure = np.zeros((3, 3))
        structure[1 + dy, 1 + dx] = 1
        graph.add_grid_edges(nodes, weights=edge, structure=structure, symmetric=False)
    graph.add_grid_tedges(nodes, switch, keep)
    graph.maxflow()
    return np.where(graph.get_grid_segments(nodes), alpha, labels)


def solve(costs, weights, neighbours, n_theta, max_cycles=4, init=None, tolerance=1e-6):
    labels = np.argmin(costs, axis=0) if init is None else init.copy()
    current = energy(labels, costs, weights, neighbours, n_theta)
    history = [current]
    for _ in range(max_cycles):
        improved = False
        for alpha in range(len(costs)):
            candidate = expand(labels, alpha, costs, weights, neighbours, n_theta)
            value = energy(candidate, costs, weights, neighbours, n_theta)
            if value < current - tolerance:
                labels, current, improved = candidate, value, True
                history.append(current)
        if not improved:
            break
    return labels, history


def segment(
    features,
    gray,
    tv,
    seeds,
    lam=0.25,
    n_theta=8,
    max_cycles=4,
    max_k=8,
    seed=0,
    reg_covar=1.0,
    hard_cost=1e6,
    sigma=None,
    neighbours=((0, 1), (1, 0), (1, 1), (1, -1)),
    init=None,
    tolerance=1e-6,
):
    """Return road labels, orientations, energy history and debug costs."""
    if n_theta < 1 or lam < 0 or max_cycles < 0:
        raise ValueError("Use n_theta >= 1, lam >= 0 and max_cycles >= 0.")
    sparse = np.where(tv["curve_points"], tv["theta"], np.nan)
    costs, probability = data_costs(
        features, sparse, seeds, n_theta, max_k, seed, reg_covar, hard_cost
    )
    weights = edge_weights(gray, tv["theta"], lam, neighbours, sigma)
    labels, history = solve(
        costs, weights, neighbours, n_theta, max_cycles, init, tolerance
    )
    road = labels >= n_theta
    orientation = np.where(road, (labels % n_theta) * np.pi / n_theta, np.nan)
    return {
        "labels": labels,
        "road": road,
        "orientation": orientation,
        "energy": history[-1],
        "energy_history": history,
        "data_costs": costs,
        "edge_weights": weights,
        "foreground_probability": probability,
        "initial_road": np.argmin(costs, axis=0) >= n_theta,
    }


def segment_intensity(features, gray, seeds, **parameters):
    tv = {"curve_points": np.zeros(gray.shape, bool), "theta": np.zeros(gray.shape)}
    return segment(features, gray, tv, seeds, n_theta=1, **parameters)
