import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np

from .map_io import compute_wall_outline_mask, load_occupancy_map


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a KDM CSV map into a heatmap image.")
    parser.add_argument("csv_map", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path. Defaults to the CSV path with a .png suffix.",
    )
    parser.add_argument(
        "--occupancy-yaml",
        help="Optional occupancy.yaml file used to mask obstacle cells or rasterize observation CSVs.",
    )
    parser.add_argument(
        "--observation-column",
        default="gas_ppm",
        help="Column name to use when the input CSV is an observations table. Default: gas_ppm",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma in meters for observations CSVs. Default: 0 (no smoothing)",
    )
    parser.add_argument(
        "--smooth-radius",
        type=float,
        default=3.0,
        help="Gaussian support radius in sigmas for smoothed observations. Default: 3.0",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap to use. Default: inferno",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for automatic color scaling. Default: 99",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Explicit upper bound for color scaling. Overrides --percentile.",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate the data grid by 180 degrees before rendering.",
    )
    parser.add_argument(
        "--flip-left-right",
        action="store_true",
        help="Flip the data grid horizontally before rendering.",
    )
    parser.add_argument(
        "--flip-up-down",
        action="store_true",
        help="Flip the data grid vertically before rendering before the final image save.",
    )
    parser.add_argument(
        "--subtract-dominant-background",
        action="store_true",
        help="Subtract the dominant rounded cell value before rendering to enhance nearly-flat KDM grids.",
    )
    parser.add_argument(
        "--deviation-mode",
        choices=("auto", "positive", "negative", "absolute"),
        default="auto",
        help="How to visualize values after subtracting the dominant background. Default: auto",
    )
    return parser.parse_args()


def observation_rows_to_grid(csv_path, occupancy, value_column):
    try:
        table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not read observations CSV '{csv_path}': {exc}") from exc

    if getattr(table, "dtype", None) is None or table.dtype.names is None:
        raise RuntimeError(
            f"Input '{csv_path}' is not a 2D matrix and does not look like a header-based observations CSV."
        )

    required = {"x", "y", value_column}
    missing = [name for name in required if name not in table.dtype.names]
    if missing:
        raise RuntimeError(
            f"Observations CSV '{csv_path}' is missing required columns: {', '.join(missing)}."
        )

    rows = np.atleast_1d(table)
    matrix = np.full((occupancy.height, occupancy.width), np.nan, dtype=float)
    sums = np.zeros((occupancy.height, occupancy.width), dtype=float)
    counts = np.zeros((occupancy.height, occupancy.width), dtype=int)

    for sample in rows:
        x = float(sample["x"])
        y = float(sample["y"])
        value = float(sample[value_column])

        col = int(np.floor((x - occupancy.origin_x) / occupancy.resolution))
        row_from_bottom = int(np.floor((y - occupancy.origin_y) / occupancy.resolution))
        row = occupancy.height - 1 - row_from_bottom

        if row < 0 or row >= occupancy.height or col < 0 or col >= occupancy.width:
            continue
        if not occupancy.free_mask[row, col]:
            continue

        sums[row, col] += value
        counts[row, col] += 1

    observed = counts > 0
    matrix[observed] = sums[observed] / counts[observed]
    return matrix


def smooth_observation_rows_to_grid(csv_path, occupancy, value_column, sigma_m, radius_sigmas):
    try:
        table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not read observations CSV '{csv_path}': {exc}") from exc

    if getattr(table, "dtype", None) is None or table.dtype.names is None:
        raise RuntimeError(
            f"Input '{csv_path}' is not a 2D matrix and does not look like a header-based observations CSV."
        )

    required = {"x", "y", value_column}
    missing = [name for name in required if name not in table.dtype.names]
    if missing:
        raise RuntimeError(
            f"Observations CSV '{csv_path}' is missing required columns: {', '.join(missing)}."
        )

    rows = np.atleast_1d(table)
    matrix = np.full((occupancy.height, occupancy.width), np.nan, dtype=float)
    numerator = np.zeros((occupancy.height, occupancy.width), dtype=float)
    denominator = np.zeros((occupancy.height, occupancy.width), dtype=float)
    free_mask = occupancy.free_mask

    sigma_cells = max(sigma_m / occupancy.resolution, 1e-6)
    radius_cells = max(1, int(np.ceil(radius_sigmas * sigma_cells)))

    for sample in rows:
        x = float(sample["x"])
        y = float(sample["y"])
        value = float(sample[value_column])

        center_col = int(np.floor((x - occupancy.origin_x) / occupancy.resolution))
        center_row_from_bottom = int(np.floor((y - occupancy.origin_y) / occupancy.resolution))
        center_row = occupancy.height - 1 - center_row_from_bottom

        if center_row < 0 or center_row >= occupancy.height or center_col < 0 or center_col >= occupancy.width:
            continue

        row_min = max(0, center_row - radius_cells)
        row_max = min(occupancy.height, center_row + radius_cells + 1)
        col_min = max(0, center_col - radius_cells)
        col_max = min(occupancy.width, center_col + radius_cells + 1)

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                if not free_mask[row, col]:
                    continue

                cell_x = occupancy.origin_x + (col + 0.5) * occupancy.resolution
                cell_y = occupancy.origin_y + (occupancy.height - row - 0.5) * occupancy.resolution
                distance_sq = (cell_x - x) ** 2 + (cell_y - y) ** 2
                weight = np.exp(-0.5 * distance_sq / (sigma_m * sigma_m))
                numerator[row, col] += weight * value
                denominator[row, col] += weight

    observed = denominator > 0
    matrix[observed] = numerator[observed] / denominator[observed]
    return matrix


def subtract_dominant_background(matrix, finite_mask, deviation_mode):
    values = np.asarray(matrix[finite_mask], dtype=float)
    rounded = np.round(values, 9)
    unique_values, counts = np.unique(rounded, return_counts=True)
    dominant_value = float(unique_values[np.argmax(counts)])

    adjusted = np.array(matrix, dtype=float, copy=True)
    adjusted[finite_mask] = adjusted[finite_mask] - dominant_value

    if deviation_mode == "auto":
        positive_peak = float(np.nanmax(adjusted[finite_mask]))
        negative_peak = float(abs(np.nanmin(adjusted[finite_mask])))
        deviation_mode = "positive" if positive_peak >= negative_peak else "negative"

    if deviation_mode == "positive":
        adjusted[finite_mask] = np.maximum(adjusted[finite_mask], 0.0)
    elif deviation_mode == "negative":
        adjusted[finite_mask] = np.maximum(-adjusted[finite_mask], 0.0)
    elif deviation_mode == "absolute":
        adjusted[finite_mask] = np.abs(adjusted[finite_mask])

    return adjusted


def transform_matrix(matrix, rotate_180=False, flip_left_right=False, flip_up_down=False):
    transformed = np.array(matrix, copy=True)
    if rotate_180:
        transformed = np.rot90(transformed, 2)
    if flip_left_right:
        transformed = np.fliplr(transformed)
    if flip_up_down:
        transformed = np.flipud(transformed)
    return transformed


def main():
    args = parse_args()
    csv_path = Path(args.csv_map)
    output_path = Path(args.output) if args.output else csv_path.with_suffix(".png")

    free_mask = None
    occupancy = None
    if args.occupancy_yaml:
        occupancy = load_occupancy_map(args.occupancy_yaml)
        free_mask = occupancy.free_mask

    try:
        matrix = np.loadtxt(csv_path, delimiter=",")
        if matrix.ndim != 2:
            raise RuntimeError
        if free_mask is not None and free_mask.shape != matrix.shape:
            raise RuntimeError(
                f"Occupancy mask shape {free_mask.shape} does not match CSV shape {matrix.shape}."
            )
    except Exception:
        if occupancy is None:
            raise RuntimeError(
                f"'{csv_path}' is not a plain 2D matrix. Provide --occupancy-yaml to rasterize an observations CSV."
            )
        if args.smooth_sigma > 0:
            matrix = smooth_observation_rows_to_grid(
                csv_path,
                occupancy,
                args.observation_column,
                float(args.smooth_sigma),
                float(args.smooth_radius),
            )
        else:
            matrix = observation_rows_to_grid(csv_path, occupancy, args.observation_column)

    matrix = transform_matrix(
        matrix,
        rotate_180=args.rotate_180,
        flip_left_right=args.flip_left_right,
        flip_up_down=args.flip_up_down,
    )

    finite_mask = np.isfinite(matrix)
    if free_mask is not None:
        finite_mask &= free_mask
    if not finite_mask.any():
        raise RuntimeError(f"No finite map values found in '{csv_path}'.")

    working = np.array(matrix, dtype=float, copy=True)
    if args.subtract_dominant_background:
        working = subtract_dominant_background(working, finite_mask, args.deviation_mode)

    vmin = float(np.nanmin(working[finite_mask]))
    vmax = float(args.vmax) if args.vmax is not None else float(np.nanpercentile(working[finite_mask], args.percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cm.get_cmap(args.cmap)(norm(np.nan_to_num(working, nan=vmin)))

    if free_mask is not None:
        rgba[~free_mask, :3] = 0.12
        rgba[~free_mask, 3] = 1.0
        wall_outline = compute_wall_outline_mask(free_mask)
        rgba[wall_outline, :3] = (0.92, 0.92, 0.92)
        rgba[wall_outline, 3] = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, np.flipud(rgba))
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
