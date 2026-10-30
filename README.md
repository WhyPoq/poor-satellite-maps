Road extraction following the stages in `paper.pdf`: Gabor responses → tensor voting → seeded graph cuts → centerlines → connected road segments. Each module uses functions; segments are dictionaries with `pts`, `theta`, `width` and `meta`.

Install and run:

```sh
python -m pip install -r requirements.txt
python draw_seeds.py satellite.png seeds.json
python pipeline.py satellite.png seeds.json --output results
python pipeline.py satellite.png seeds.json --output results --config config.example.json
```

In the seed window, paint roads with the left mouse button and background with the right. Red means road, blue means background. `C` clears, `S` saves and exits, and `Esc` cancels. Coordinates refer to the original image, even when the window is resized. Seeds are plain JSON:

```json
{"fg": [[20, 30], [21, 30]], "bg": [[5, 5], [6, 5]]}
```

Each point is `[x, y]`. Mark representative colors for both classes, including shadows when relevant. Seed coordinates must be inside the image and cannot belong to both classes.

Opening the seed tool with an existing output file reloads those seeds so you can extend them. In `italy2`, include road surfaces in sunlight and shadow as foreground, and several roof colors, vegetation, soil, and building shadows as background. The color model cannot reliably distinguish roads and roofs of the same color from the existing few seed locations alone.

From Python:

```python
from pipeline import run_pipeline

result = run_pipeline(
    "satellite.png", "seeds.json", output="results",
    tensor_params={"sigma": 12, "n": 8},
    segmentation_params={"lam": 0.25, "max_cycles": 4},
    centerline_params={"widths": [6, 8, 10, 12]},
    network_params={"keep_longest": False},
)
mask = result["road"]
```

The function also accepts a BGR `uint8` OpenCV image and a seed dictionary. It returns the road mask, road orientations in radians, final segments, and all stage results. Omit `output` to keep everything in memory. Each parameter dictionary is passed to the corresponding module's function; their signatures list the available settings. Widths, distances and voting sigma are in pixels; angles are in radians. Optional smoothing uses `smooth_params={}`; optional line simplification uses `simplify_tol=1.5`.

With an output folder, every stage saves exact arrays in `arrays.npz`, normalized PNG previews, and JSON metadata for segments and energy history. Gabor and graph-cut label costs have a preview per orientation/label. `09_result/road.png` is the binary segmentation and `09_result/overlay.png` shows roads in red and final centerlines in yellow. Load arrays with `np.load("results/03_tensor/arrays.npz")`; preview brightness is normalized independently and is only for inspection. Debug output includes large per-label arrays, so use a crop first for large images. No resizing is performed.

In `04_segmentation`, `foreground_probability.png` shows the color model before orientation/smoothing, and `initial_road.png` shows the mask before graph cuts. If these already select roofs or miss shadowed roads, improve the seeds first. Label-cost previews clip the top 1% of values to stop hard seed penalties from hiding all other costs; NPZ arrays retain the original values.

The refactor retains the existing implementation's approximations: 2D steerable tensor voting with stick-only votes by default, normalized class likelihoods, and truncated alpha-expansion moves accepted only when energy decreases. It is not an exact reproduction of every detail of the paper. Contrast uncertainty is estimated automatically unless `segmentation_params.sigma` is set. Disconnected networks are retained by default; `keep_longest=true` explicitly enables pruning. Proposed connections must lie mostly on the road mask (`network_params.min_support`, default 0.7). Interactive refinement functions remain available in `network.py`.

For `italy2`, a separate config increases `lam` from 0.25 to 2 to reduce speckling. This is a starting point, not a measured accuracy improvement; stronger smoothing can also remove narrow roads. It leaves the core segmentation defaults unchanged:

```sh
python pipeline.py data/italy2.png seeds/italy2.json --output results_checked --config config.italy2.json
python draw_seeds.py data/italy2.png seeds/italy2.json
```

Run the nine checks with `python test_pipeline.py`. They cover a synthetic road, a blank image, brightness invariance, centerline masking, supported network connections, tensor directions, invalid seeds, binary graph-cut energy against exhaustive search, and seed saving with simulated mouse input. Synthetic checks establish specific behavior; the satellite image has no reference road mask for measuring accuracy.
