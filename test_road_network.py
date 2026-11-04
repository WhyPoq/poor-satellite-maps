import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

import road_network
from pipeline import run_pipeline
from road_map import BLOCK, ROAD, render_road_map


def extract(mask, **params):
    defaults = dict(widths=[8, 12, 16], n_orientations=4, curve_source="mask")
    zeros = np.zeros(mask.shape, np.float32)
    image = np.zeros((*mask.shape, 3), np.uint8)
    return road_network.extract(image, zeros, zeros, mask, **{**defaults, **params})


def test_clean_mask_keeps_roads_and_drops_speckle():
    mask = np.zeros((120, 200), bool)
    mask[50:62, :] = True  # 12 px road
    mask[10:12, 20:180] = True  # 2 px strip
    mask[90:93, 30:33] = True  # speckle
    mask[95:115, 150:170] = True  # wide but short blob, as on a roof
    cleaned = road_network.clean_mask(mask, 10, 20)
    assert cleaned[51:61, 5:195].all()
    assert not (cleaned[:40].any() or cleaned[80:].any())


def test_merge_parallel_joins_lane_bands_and_keeps_separate_roads():
    lanes = [np.array([[0, 50], [100, 50]]), np.array([[80, 58], [180, 58.5]])]
    lines, widths = road_network.merge_parallel(lanes, lambda line: 10)
    assert len(lines) == 1 and widths == [10]
    np.testing.assert_allclose(lines[0][:, 0], [0, 180], atol=1)
    assert 53 < lines[0][:, 1].mean() < 55
    apart = [
        np.array([[0, 50], [100, 50]]),
        np.array([[0, 80], [100, 80]]),
        np.array([[50, 0], [50, 100]]),
    ]
    assert len(road_network.merge_parallel(apart, lambda line: 10)[0]) == 3


def test_mask_width_measures_road_beyond_tested_widths():
    mask = np.zeros((120, 200), np.uint8)
    mask[30:90, :] = 1  # 60 px road
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    assert abs(road_network.mask_width([[10, 60], [190, 60]], distance) - 60) <= 2
    assert road_network.mask_width([[10, 5], [190, 5]], distance) == 0


def test_extracts_known_road_width_direction_and_polygon():
    mask = np.zeros((64, 96), bool)
    mask[26:38, 8:88] = True
    calls = []
    result = extract(mask, progress=calls.append)
    assert result["centreline_count"] == 1
    assert abs(result["centrelines"][0]["width"] - 12) < 1e-6
    assert abs(result["total_length_px"] - 79) < 1e-6
    assert result["polygon_area_px2"] > 900
    assert result["network_mask"][32, 48] and not result["network_mask"][0, 0]
    assert result["centreline_mask"].any()
    assert any("filters" in call for call in calls)


def test_empty_road_mask_returns_empty_outputs_and_warning():
    result = extract(np.zeros((40, 40), bool))
    assert result["centreline_count"] == 0 and result["total_length_px"] == 0
    assert not result["network_mask"].any() and not result["centreline_mask"].any()
    assert result["warnings"]


def test_longest_component_choice():
    mask = np.zeros((96, 128), bool)
    mask[15:27, 8:100] = True
    mask[65:77, 8:65] = True
    longest = extract(mask)
    everything = extract(mask, keep_longest=False)
    assert everything["centreline_count"] > longest["centreline_count"]
    assert everything["total_length_px"] > longest["total_length_px"]


def test_refinement_actions():
    mask = np.zeros((64, 96), bool)
    mask[26:38, 8:88] = True
    result = extract(mask, actions=[{"action": "delete", "id": 0}])
    assert result["centreline_count"] == 0
    added = extract(mask, actions=[{"action": "add", "points": [[48, 0], [48, 20]]}])
    assert added["centreline_count"] == 2


def test_invalid_settings():
    mask = np.ones((20, 20), bool)
    for params in (
        {"widths": []},
        {"widths": [0, 4]},
        {"n_orientations": 4.5},
        {"curve_source": "unknown"},
        {"sigma_along": float("nan")},
    ):
        try:
            extract(mask, **params)
        except ValueError:
            continue
        raise AssertionError(f"accepted {params}")


def test_road_map_draws_lines_and_only_wide_road_areas():
    original = np.zeros((100, 160, 3), np.uint8)
    lines = [{"points": [[0, 20], [160, 20]], "width": 8}]
    area = np.zeros((100, 160), bool)
    area[40:90, 20:70] = True  # plaza, much wider than the 8 px road
    area[60:63, 100:160] = True  # thin fringe of the mask
    image = render_road_map(lines, original, road_area=area)
    np.testing.assert_array_equal(image[20, 80], ROAD)
    np.testing.assert_array_equal(image[65, 45], ROAD)
    np.testing.assert_array_equal(image[61, 130], BLOCK)
    np.testing.assert_array_equal(image[5, 80], BLOCK)


def test_pipeline_stage():
    image = np.full((64, 96, 3), (35, 65, 35), np.uint8)
    image[26:38] = 190
    seeds = {"fg": [[10, 31]], "bg": [[10, 5]]}
    with TemporaryDirectory() as folder:
        result = run_pipeline(
            image,
            seeds,
            folder,
            extraction_params={"widths": [8, 12, 16], "n_orientations": 4},
        )
        network = result["road_network"]
        assert network["centreline_count"] >= 1
        stage = Path(folder) / "05_road_network"
        road_map = cv2.imread(str(stage / "road_map.png"))
        assert road_map.shape == image.shape
        metadata = json.loads((stage / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["road_network"]["type"] in ("Polygon", "MultiPolygon")
        assert len(metadata["centrelines"]) == network["centreline_count"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
