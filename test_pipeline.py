import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

import centerline
import draw_seeds
import gabor
import network
import segmentation
import tensor_voting
from pipeline import run_pipeline


def test_pipeline():
    image = np.full((64, 64, 3), (35, 65, 35), np.uint8)
    image[27:37] = 190
    seeds = {"fg": [[10, 30]], "bg": [[10, 5]]}
    with TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.png"
        seeds_path = Path(folder) / "seeds.json"
        cv2.imwrite(str(image_path), image)
        seeds_path.write_text(json.dumps(seeds), encoding="utf-8")
        result = run_pipeline(
            image_path,
            seeds_path,
            folder,
            smooth_params={"iters": 2},
            simplify_tol=1.5,
        )
        assert result["road"][30].all()
        assert not result["road"][:10].any()
        assert len(result["segments"]) >= 1
        with np.load(Path(folder) / "04_segmentation" / "arrays.npz") as saved:
            np.testing.assert_array_equal(saved["road"], result["road"])
            assert saved["data_costs"].shape == (16, 64, 64)
        for stage in result["stages"]:
            assert (Path(folder) / stage).is_dir()
        overlay = cv2.imread(str(Path(folder) / "09_result" / "overlay.png"))
        assert overlay.shape == image.shape
        history = result["stages"]["04_segmentation"]["energy_history"]
        assert np.all(np.diff(history) < 0)
        segments = result["segments"]
        parts = network.split(segments[0], segments[0]["pts"][0])
        assert all(len(part["pts"]) >= 2 for part in parts)


def test_uniform_image():
    result = run_pipeline(
        np.zeros((16, 16, 3), np.uint8),
        {"fg": [[8, 8]], "bg": [[0, 0]]},
        gabor_params={"n_orient": 4, "n_scale": 2},
        tensor_params={"sigma": 2, "n": 2},
        segmentation_params={"n_theta": 2},
        centerline_params={"widths": [2, 4], "n_orient": 4},
    )
    assert result["road"][8, 8] and not result["road"][0, 0]
    assert result["segments"] == []
    assert np.isfinite(result["stages"]["04_segmentation"]["energy"])


def test_gabor_brightness():
    for brightness in (0.0, 0.2, 1.0):
        response = gabor.gabor_bank(np.full((32, 32), brightness, np.float32))
        assert not response.any(), "Flat brightness must not create edges."
    image = np.zeros((32, 32), np.float32)
    image[12:20] = 0.5
    np.testing.assert_allclose(
        gabor.gabor_bank(image), gabor.gabor_bank(image + 0.25), atol=1e-6
    )


def test_centerline_stays_on_road():
    road = np.zeros((32, 32), bool)
    road[12:20] = True
    curves = np.zeros_like(road)
    curves[[12, 19]] = True
    response, _, _ = centerline.filter_maps(
        curves, np.zeros(road.shape), road, widths=[8]
    )
    assert not response[~road].any()
    assert response[road].max() > 0


def test_network_connections():
    a = network.from_points([[2, 8], [10, 8]], 4)
    b = network.from_points([[20, 8], [28, 8]], 4)
    road = np.zeros((16, 32), bool)
    road[6:11, 2:11] = True
    road[6:11, 20:29] = True
    assert len(network.snap_and_prune([a, b], road=road)) == 2
    road[6:11, 10:21] = True
    assert len(network.snap_and_prune([a, b], road=road)) == 1
    assert network.snap_and_prune([network.from_points([[2, 8]], 4)]) == []


def test_tensor_directions():
    for angle, points in ((0, [(16, 8), (16, 24)]), (np.pi / 4, [(8, 8), (24, 24)])):
        stick = np.zeros((33, 33))
        for y, x in points:
            stick[y, x] = 1
        saliency, ball, theta = tensor_voting.decompose(
            *tensor_voting.vote(
                stick, np.zeros_like(stick), np.full(stick.shape, angle), sigma=12
            )
        )
        assert saliency[16, 16] > ball[16, 16]
        assert segmentation.orient_dist(theta[16, 16], angle) < 1e-6


def test_invalid_seeds():
    for seeds in (
        {"fg": [], "bg": [[0, 0]]},
        {"fg": [[0, 0]], "bg": [[0, 0]]},
        {"fg": [[-1, 0]], "bg": [[0, 0]]},
        {"fg": [[20, 0]], "bg": [[0, 0]]},
        {"fg": [[1.5, 0]], "bg": [[0, 0]]},
    ):
        try:
            segmentation.validate_seeds(seeds, (10, 10))
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid seeds: {seeds}")


def test_graph_cut():
    rng = np.random.default_rng(3)
    neighbours = ((0, 1), (1, 0), (1, 1), (1, -1))
    for _ in range(5):
        costs = rng.random((2, 2, 2))
        weights = segmentation.edge_weights(rng.random((2, 2)), np.zeros((2, 2)))
        labels, history = segmentation.solve(costs, weights, neighbours, 1)
        exact = min(
            segmentation.energy(
                np.array(values).reshape(2, 2), costs, weights, neighbours, 1
            )
            for values in itertools.product(range(2), repeat=4)
        )
        np.testing.assert_allclose(history[-1], exact)
        assert labels.shape == (2, 2)


def test_seed_drawing():
    with TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.png"
        output = Path(folder) / "seeds.json"
        cv2.imwrite(str(image_path), np.zeros((10, 10, 3), np.uint8))
        output.write_text(json.dumps({"fg": [[1, 1]], "bg": [[8, 1]]}))

        def paint(_, callback):
            callback(cv2.EVENT_LBUTTONDOWN, 3, 4, cv2.EVENT_FLAG_LBUTTON, None)
            callback(cv2.EVENT_LBUTTONUP, 3, 4, 0, None)
            callback(cv2.EVENT_RBUTTONDOWN, 7, 8, cv2.EVENT_FLAG_RBUTTON, None)
            callback(cv2.EVENT_RBUTTONUP, 7, 8, 0, None)

        with patch.multiple(
            cv2,
            namedWindow=lambda *a: None,
            imshow=lambda *a: None,
            setMouseCallback=paint,
            waitKey=lambda *a: ord("s"),
            destroyAllWindows=lambda: None,
        ):
            draw_seeds.draw_seeds(image_path, output, radius=0)
        assert json.loads(output.read_text()) == {
            "fg": [[1, 1], [3, 4]],
            "bg": [[8, 1], [7, 8]],
        }


if __name__ == "__main__":
    for test in (
        test_pipeline,
        test_uniform_image,
        test_gabor_brightness,
        test_centerline_stays_on_road,
        test_network_connections,
        test_tensor_directions,
        test_invalid_seeds,
        test_graph_cut,
        test_seed_drawing,
    ):
        test()
        print(f"{test.__name__}: passed")
