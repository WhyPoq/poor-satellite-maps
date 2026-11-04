Road extraction following the stages in `paper.pdf`: Gabor responses → tensor voting → seeded graph cuts → road network extraction and modelling. Each module uses functions.

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
    extraction_params={"widths": range(10, 61, 4), "keep_longest": False},
)
mask = result["road"]
network = result["road_network"]
```

The function also accepts a BGR `uint8` OpenCV image and a seed dictionary. It returns the road mask, road orientations in radians, the road network, and all stage results. Omit `output` to keep everything in memory. Each parameter dictionary is passed to the corresponding module's function; their signatures list the available settings. Widths, distances and voting sigma are in pixels; angles are in radians.

With an output folder, every stage saves exact arrays in `arrays.npz`, normalized PNG previews, and JSON metadata for the network and energy history. Gabor and graph-cut label costs have a preview per orientation/label. `06_result/road.png` is the binary segmentation and `06_result/overlay.png` shows roads in red and final centrelines in yellow. Load arrays with `np.load("results/03_tensor/arrays.npz")`; preview brightness is normalized independently and is only for inspection. Debug output includes large per-label arrays, so use a crop first for large images. No resizing is performed.

In `04_segmentation`, `foreground_probability.png` shows the color model before orientation/smoothing, and `initial_road.png` shows the mask before graph cuts. If these already select roofs or miss shadowed roads, improve the seeds first. Label-cost previews clip the top 1% of values to stop hard seed penalties from hiding all other costs; NPZ arrays retain the original values.

The refactor retains the existing implementation's approximations: 2D steerable tensor voting with stick-only votes by default, normalized class likelihoods, and truncated alpha-expansion moves accepted only when energy decreases. It is not an exact reproduction of every detail of the paper. Contrast uncertainty is estimated automatically unless `segmentation_params.sigma` is set.

For `italy2`, a separate config increases `lam` from 0.25 to 2 to reduce speckling and keeps disconnected networks. This is a starting point, not a measured accuracy improvement; stronger smoothing can also remove narrow roads. It leaves the core segmentation defaults unchanged:

```sh
python pipeline.py data/italy2.png seeds/italy2.json --output results_checked --config config.italy2.json
python draw_seeds.py data/italy2.png seeds/italy2.json
```

### Road network extraction and modelling

`road_network.py` and `road_map.py` turn the road mask into a modelled network (Section 6). `extraction_params` is passed to `road_network.extract`. The stage optionally cleans the road mask with oriented line openings (`clean=True`), runs the bimodal road-side and single-mode road-area centreline filters over `widths` and `n_orientations` (`sigma_across=1.5`, `sigma_along=6`), and votes their responses in an iterative Hough transform. `keep_longest=True` (default) keeps only the longest connected network. Segments are cut where they leave the mask, near-parallel lane bands are merged into one centreline measured from the mask, and ends are extended to nearby junctions. Road sides are tracked from each centreline (Eq. 21) and unioned into a polygon footprint. `curve_source="tensor"` uses the Stage 1 curve map as road-side evidence; `"mask"` uses the mask edges. Refinement actions 1–7 (seed tracking, add, edit, Hermite merge, delete, snake smoothing, point reduction) can be applied with `actions=[...]`; the format is in `road_network.refine`.

`05_road_network` saves the filter magnitude, orientation and width, the cleaned mask, centreline and polygon rasters, an overlay (footprint in green, centrelines in yellow) and `road_map.png`, a street-map rendering with cream blocks, blue-grey roads, white kerbs and vegetation from the input as parks and medians. Enclosed holes in roads and parks smaller than `HOLE_AREA` (10) squared median road widths are always filled; larger enclosed regions stay, as they are city blocks or islands. `schematic_map.png` draws only the simplified centrelines, with three stroke widths for road classes, on a plain background. `metadata.json` holds the centrelines (`id`, `points`, `width`), the footprint as pixel-coordinate GeoJSON, counts, length, area and warnings. Coordinates are `(x, y)` image pixels with y downwards, not georeferenced. With a large `max(widths)` the filters are slow on full images; use a crop first.

Run the checks with `python test_pipeline.py` and `python test_road_network.py`. The eight pipeline checks cover a synthetic road, a blank image, brightness invariance, the centreline filter peak, tensor directions, invalid seeds, binary graph-cut energy against exhaustive search, and seed saving with simulated mouse input; the road network checks cover mask cleaning, lane merging, mask widths, a known road, an empty mask, longest-network pruning, refinement actions, invalid settings, map rendering and the saved stage. Synthetic checks establish specific behavior; the satellite image has no reference road mask for measuring accuracy.
