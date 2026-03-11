import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from gaden_msgs.srv import GasPosition
from rclpy.node import Node

from .map_io import compute_wall_outline_mask, load_occupancy_map


class GroundTruthExporter(Node):
    def __init__(self, service_name):
        super().__init__("ground_truth_exporter")
        self.client = self.create_client(GasPosition, service_name)

    def wait_for_service(self):
        while rclpy.ok() and not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for gas service '{self.client.srv_name}'...")

    def sample(self, xs, ys, zs):
        request = GasPosition.Request()
        request.x = [float(value) for value in xs]
        request.y = [float(value) for value in ys]
        request.z = [float(value) for value in zs]

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError("Gas service call failed.")
        return future.result()


def parse_args():
    parser = argparse.ArgumentParser(description="Export a 2D ground-truth gas slice from the GADEN player service.")
    parser.add_argument("--occupancy-yaml", required=True, help="Path to the occupancy.yaml file used for the map.")
    parser.add_argument("--z", type=float, default=0.5, help="Slice height in meters. Default: 0.5")
    parser.add_argument("--service", default="/odor_value", help="Gas service name. Default: /odor_value")
    parser.add_argument("--gas-type", help="Gas type name to export. Default: first gas returned by the service.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Number of cells to sample per service call.")
    parser.add_argument("--output-csv", help="Output CSV path. Defaults to <occupancy dir>/ground_truth_z_<z>.csv")
    parser.add_argument("--output-png", help="Optional output PNG path. Defaults to the CSV path with .png suffix.")
    parser.add_argument("--cmap", default="inferno", help="Matplotlib colormap for the PNG. Default: inferno")
    parser.add_argument("--percentile", type=float, default=99.0, help="Upper percentile for color scaling. Default: 99")
    return parser.parse_args()


def cell_centers(map_metadata, free_mask):
    xs = []
    ys = []
    cells = []
    for row in range(map_metadata.height):
        for col in range(map_metadata.width):
            if not free_mask[row, col]:
                continue
            x = map_metadata.origin_x + (col + 0.5) * map_metadata.resolution
            y = map_metadata.origin_y + (map_metadata.height - row - 0.5) * map_metadata.resolution
            cells.append((row, col))
            xs.append(x)
            ys.append(y)
    return cells, xs, ys


def render_png(matrix, free_mask, output_path, cmap_name, percentile):
    finite_mask = np.isfinite(matrix) & free_mask
    if not finite_mask.any():
        raise RuntimeError("Ground-truth matrix contains no finite free-space values.")

    vmin = float(np.nanmin(matrix[finite_mask]))
    vmax = float(np.nanpercentile(matrix[finite_mask], percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cm.get_cmap(cmap_name)(norm(np.nan_to_num(matrix, nan=vmin)))
    rgba[~free_mask, :3] = 0.12
    rgba[~free_mask, 3] = 1.0
    wall_outline = compute_wall_outline_mask(free_mask)
    rgba[wall_outline, :3] = (0.92, 0.92, 0.92)
    rgba[wall_outline, 3] = 1.0
    plt.imsave(output_path, np.flipud(rgba))


def main():
    args = parse_args()
    occupancy_yaml = Path(args.occupancy_yaml)
    map_metadata = load_occupancy_map(occupancy_yaml)
    free_mask = map_metadata.free_mask

    default_name = f"ground_truth_z_{str(args.z).replace('.', 'p')}.csv"
    output_csv = Path(args.output_csv) if args.output_csv else occupancy_yaml.parent / default_name
    output_png = Path(args.output_png) if args.output_png else output_csv.with_suffix(".png")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cells, xs, ys = cell_centers(map_metadata, free_mask)
    zs = [args.z] * len(xs)

    matrix = np.full((map_metadata.height, map_metadata.width), np.nan, dtype=float)

    rclpy.init()
    node = GroundTruthExporter(args.service)
    try:
        node.wait_for_service()

        selected_gas_index = None
        for start in range(0, len(xs), args.chunk_size):
            stop = min(len(xs), start + args.chunk_size)
            response = node.sample(xs[start:stop], ys[start:stop], zs[start:stop])

            if selected_gas_index is None:
                if not response.gas_type:
                    raise RuntimeError("Gas service returned no gas types.")
                if args.gas_type:
                    try:
                        selected_gas_index = list(response.gas_type).index(args.gas_type)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Gas type '{args.gas_type}' not found. Available types: {', '.join(response.gas_type)}"
                        ) from exc
                else:
                    selected_gas_index = 0
                node.get_logger().info(f"Exporting gas type '{response.gas_type[selected_gas_index]}'.")

            for local_index, gas_cell in enumerate(response.positions):
                row, col = cells[start + local_index]
                matrix[row, col] = float(gas_cell.concentration[selected_gas_index])
    finally:
        node.destroy_node()
        rclpy.shutdown()

    np.savetxt(output_csv, matrix, delimiter=",")
    render_png(matrix, free_mask, output_png, args.cmap, args.percentile)
    print(f"Saved ground-truth CSV to {output_csv}")
    print(f"Saved ground-truth PNG to {output_png}")


if __name__ == "__main__":
    main()
