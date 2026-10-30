import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import centerline
import gabor
import network
import segmentation
import tensor_voting


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def preview(array):
    """Scale a debug map to 8 bits; exact values stay in the NPZ file."""
    if array.dtype == np.uint8:
        return array
    if array.dtype == bool:
        return array.astype(np.uint8) * 255
    finite = np.isfinite(array)
    image = np.zeros(array.shape, np.uint8)
    if finite.any():
        low, high = array[finite].min(), array[finite].max()
        if high > low:
            image[finite] = ((array[finite] - low) / (high - low) * 255).astype(
                np.uint8
            )
        elif high > 0:
            image[finite] = 255
    return image


def json_value(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def save_stage(output, name, values):
    """Save arrays losslessly, with previews and small metadata alongside."""
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value for key, value in values.items() if isinstance(value, np.ndarray)
    }
    if "edge_weights" in values:
        arrays.update(
            {
                f"edge_weight_{i}": value
                for i, value in enumerate(values["edge_weights"])
            }
        )
    if arrays:
        np.savez_compressed(folder / "arrays.npz", **arrays)
    for key, array in arrays.items():
        if key == "data_costs":
            array = np.minimum(array, np.percentile(array, 99))
        if array.ndim == 2 or key in ("image", "overlay"):
            display = array.astype(float) if key == "seeds" else array
            cv2.imwrite(str(folder / f"{key}.png"), preview(display))
        elif array.ndim == 3:
            for i, layer in enumerate(array):
                cv2.imwrite(str(folder / f"{key}_{i:02}.png"), preview(layer))
    metadata = {
        key: value
        for key, value in values.items()
        if key not in arrays and key != "edge_weights"
    }
    if metadata:
        (folder / "metadata.json").write_text(
            json.dumps(metadata, default=json_value, indent=2), encoding="utf-8"
        )


def run_pipeline(
    image,
    seeds,
    output=None,
    gabor_params=None,
    tensor_params=None,
    segmentation_params=None,
    centerline_params=None,
    network_params=None,
    smooth_params=None,
    simplify_tol=None,
):
    """Accept a BGR uint8 image (or path) and seed dict (or JSON path)."""
    if isinstance(image, (str, Path)):
        image = read_image(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Use a uint8 BGR image with three channels.")
    if min(image.shape[:2]) < 2:
        raise ValueError("Image must be at least 2 by 2 pixels.")
    if isinstance(seeds, (str, Path)):
        seeds = json.loads(Path(seeds).read_text(encoding="utf-8"))
    seeds = segmentation.validate_seeds(seeds, image.shape[:2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    seed_map = np.zeros(gray.shape, np.uint8)
    for name, label in (("bg", 1), ("fg", 2)):
        x, y = seeds[name].T
        seed_map[y, x] = label
    stages = {}

    def record(name, values):
        stages[name] = values
        if output is not None:
            save_stage(output, name, values)

    record(
        "01_input",
        {"image": image, "gray": gray, "seeds": seed_map, "seed_points": seeds},
    )
    responses = gabor.gabor_bank(gray, **(gabor_params or {}))
    record("02_gabor", {"responses": responses})
    voting = tensor_voting.infer(responses, **(tensor_params or {}))
    record("03_tensor", voting)
    segmented = segmentation.segment(
        image, gray, voting, seeds, **(segmentation_params or {})
    )
    record("04_segmentation", segmented)
    lines = centerline.extract(
        voting["curve_points"],
        voting["theta"],
        segmented["road"],
        **(centerline_params or {}),
    )
    record("05_centerline", lines)
    context = {**voting, **segmented, **lines, "theta": voting["theta"]}
    segments = network.snap_and_prune(
        lines["segments"], road=segmented["road"], **(network_params or {})
    )
    record("06_network", {"segments": segments})
    if smooth_params is not None:
        segments = [
            network.smooth(context, segment, **smooth_params) for segment in segments
        ]
        record("07_smoothed", {"segments": segments})
    if simplify_tol is not None:
        segments = [network.simplify(segment, simplify_tol) for segment in segments]
        record("08_simplified", {"segments": segments})
    overlay = image.copy()
    road = segmented["road"]
    overlay[road] = (0.5 * overlay[road] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for segment in segments:
        points = np.round(segment["pts"]).astype(np.int32)
        cv2.polylines(overlay, [points], False, (0, 255, 255), 1)
    record("09_result", {"road": road, "overlay": overlay, "segments": segments})
    return {
        "road": road,
        "orientation": segmented["orientation"],
        "segments": segments,
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Road segmentation with intermediate debug outputs."
    )
    parser.add_argument("image")
    parser.add_argument("seeds")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--config", help="JSON containing run_pipeline parameter dictionaries"
    )
    args = parser.parse_args()
    config = (
        json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    )
    result = run_pipeline(args.image, args.seeds, args.output, **config)
    print(f"Saved results to {args.output}; {result['road'].sum()} road pixels.")


if __name__ == "__main__":
    main()
