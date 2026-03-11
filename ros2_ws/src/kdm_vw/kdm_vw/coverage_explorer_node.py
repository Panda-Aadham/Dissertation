import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

from .map_io import load_occupancy_map


class CoverageExplorerNode(Node):
    def __init__(self):
        super().__init__("coverage_explorer")

        self.declare_parameter("occupancy_yaml", "")
        self.declare_parameter("pose_topic", "ground_truth")
        self.declare_parameter("goal_topic", "goal_pose")
        self.declare_parameter("waypoint_spacing", 1.0)
        self.declare_parameter("clearance", 0.3)
        self.declare_parameter("goal_timeout", 45.0)
        self.declare_parameter("retry_limit", 2)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("start_delay", 2.0)
        self.declare_parameter("shutdown_on_complete", True)
        self.declare_parameter("shutdown_delay", 2.0)

        occupancy_yaml = self.get_parameter("occupancy_yaml").value
        if not occupancy_yaml:
            raise RuntimeError("The 'occupancy_yaml' parameter must point to a VGR occupancy yaml file.")

        self.map_metadata = load_occupancy_map(occupancy_yaml)
        self.free_mask = self.map_metadata.free_mask
        self.latest_pose = None
        self.active_goal = None
        self.active_goal_handle = None
        self.active_goal_sent_time = None
        self.server_ready = False
        self.goal_retry_counts = {}

        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)
        self.retry_limit = int(self.get_parameter("retry_limit").value)
        self.start_delay = float(self.get_parameter("start_delay").value)
        self.shutdown_on_complete = bool(self.get_parameter("shutdown_on_complete").value)
        self.shutdown_delay = float(self.get_parameter("shutdown_delay").value)
        self.started_time = float(self.get_clock().now().nanoseconds) / 1e9
        self.completion_announced = False
        self.completion_timer = None

        self.waypoints = self.build_waypoints(
            spacing=float(self.get_parameter("waypoint_spacing").value),
            clearance=float(self.get_parameter("clearance").value),
        )
        self.remaining_waypoints = set(range(len(self.waypoints)))

        self.goal_pub = self.create_publisher(PoseStamped, self.get_parameter("goal_topic").value, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10,
        )

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.create_timer(1.0, self.check_navigation_server)
        self.create_timer(0.5, self.control_loop)

        self.get_logger().info(
            f"Coverage explorer ready with {len(self.waypoints)} waypoints from map '{occupancy_yaml}'."
        )

    def pose_callback(self, msg):
        self.latest_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

    def check_navigation_server(self):
        if self.server_ready:
            return
        self.server_ready = self.nav_client.wait_for_server(timeout_sec=0.0)
        if self.server_ready:
            self.get_logger().info("Connected to Nav2 navigate_to_pose action server.")

    def control_loop(self):
        if not self.server_ready or self.latest_pose is None:
            return

        now_seconds = float(self.get_clock().now().nanoseconds) / 1e9
        if now_seconds - self.started_time < self.start_delay:
            return

        if self.active_goal is not None:
            if self.distance(self.latest_pose, self.active_goal) <= self.goal_tolerance:
                self.complete_active_goal("reached")
                return
            if (
                self.active_goal_sent_time is not None
                and now_seconds - self.active_goal_sent_time > self.goal_timeout
            ):
                self.fail_active_goal("timed out")
            return

        next_index = self.choose_next_waypoint()
        if next_index is None:
            self.handle_completion()
            return

        self.send_goal(next_index)

    def choose_next_waypoint(self):
        if not self.remaining_waypoints or self.latest_pose is None:
            return None

        return min(
            self.remaining_waypoints,
            key=lambda index: self.distance(self.latest_pose, self.waypoints[index]),
        )

    def send_goal(self, waypoint_index):
        goal_xy = self.waypoints[waypoint_index]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_xy[0])
        goal_msg.pose.pose.position.y = float(goal_xy[1])
        goal_msg.pose.pose.orientation.w = 1.0

        self.publish_goal_pose(goal_msg.pose)

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda result, idx=waypoint_index, goal=goal_xy: self.goal_response_callback(result, idx, goal))
        self.active_goal = goal_xy
        self.active_goal_sent_time = float(self.get_clock().now().nanoseconds) / 1e9
        self.get_logger().info(f"Sent exploration goal to ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}).")

    def goal_response_callback(self, future, waypoint_index, goal_xy):
        try:
            goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive ROS callback handling
            self.get_logger().warn(f"Failed to send goal {goal_xy}: {exc}")
            self.fail_active_goal("send failure")
            return

        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) was rejected by Nav2.")
            self.fail_active_goal("rejected")
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda result, idx=waypoint_index, goal=goal_xy: self.goal_result_callback(result, idx, goal))

    def goal_result_callback(self, future, waypoint_index, goal_xy):
        if self.active_goal != goal_xy:
            return

        try:
            wrapped_result = future.result()
            status = wrapped_result.status
        except Exception as exc:  # pragma: no cover - defensive ROS callback handling
            self.get_logger().warn(f"Goal result handling failed for {goal_xy}: {exc}")
            self.fail_active_goal("result failure")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.complete_active_goal("succeeded")
        else:
            self.get_logger().warn(f"Goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) ended with Nav2 status {status}.")
            self.fail_active_goal(f"status {status}")

    def complete_active_goal(self, reason):
        if self.active_goal is None:
            return

        goal_index = self.find_goal_index(self.active_goal)
        if goal_index is not None:
            self.remaining_waypoints.discard(goal_index)
            self.goal_retry_counts.pop(goal_index, None)
        self.get_logger().info(
            f"Completed exploration goal at ({self.active_goal[0]:.2f}, {self.active_goal[1]:.2f}) via {reason}. "
            f"{len(self.remaining_waypoints)} waypoints remain."
        )
        self.clear_active_goal()

    def fail_active_goal(self, reason):
        if self.active_goal is None:
            return

        goal_index = self.find_goal_index(self.active_goal)
        if goal_index is not None:
            self.goal_retry_counts[goal_index] = self.goal_retry_counts.get(goal_index, 0) + 1
            if self.goal_retry_counts[goal_index] > self.retry_limit:
                self.remaining_waypoints.discard(goal_index)
                self.get_logger().warn(
                    f"Dropping exploration goal at ({self.active_goal[0]:.2f}, {self.active_goal[1]:.2f}) after {reason}. "
                    f"{len(self.remaining_waypoints)} waypoints remain."
                )
            else:
                self.get_logger().warn(
                    f"Will retry exploration goal at ({self.active_goal[0]:.2f}, {self.active_goal[1]:.2f}) after {reason}. "
                    f"Retry {self.goal_retry_counts[goal_index]}/{self.retry_limit}."
                )

        if self.active_goal_handle is not None:
            self.active_goal_handle.cancel_goal_async()
        self.clear_active_goal()

    def clear_active_goal(self):
        self.active_goal = None
        self.active_goal_handle = None
        self.active_goal_sent_time = None

    def find_goal_index(self, goal_xy):
        for index, waypoint in enumerate(self.waypoints):
            if waypoint == goal_xy:
                return index
        return None

    def publish_goal_pose(self, pose):
        msg = PoseStamped()
        msg.header = pose.header
        msg.pose = pose.pose
        self.goal_pub.publish(msg)

    def handle_completion(self):
        if self.completion_announced:
            return

        self.completion_announced = True
        self.get_logger().info("Coverage explorer has no remaining waypoints to visit.")
        if self.shutdown_on_complete:
            self.get_logger().info(
                f"Coverage explorer is complete. Shutting down this process in {self.shutdown_delay:.1f} seconds."
            )
            self.completion_timer = self.create_timer(self.shutdown_delay, self.request_shutdown)

    def request_shutdown(self):
        if self.completion_timer is not None:
            self.completion_timer.cancel()
            self.completion_timer = None
        self.get_logger().info("Coverage explorer finished. Requesting shutdown.")
        rclpy.shutdown()
        raise SystemExit(0)

    def build_waypoints(self, spacing, clearance):
        resolution = float(self.map_metadata.resolution)
        step_cells = max(1, int(round(spacing / resolution)))
        clearance_cells = max(0, int(round(clearance / resolution)))
        eligible = self.compute_clearance_mask(clearance_cells)

        waypoints = []
        seen_cells = set()
        search_radius = max(1, step_cells // 2)
        row_indices = list(range(step_cells // 2, self.map_metadata.height, step_cells))
        for row_number, row in enumerate(row_indices):
            column_indices = list(range(step_cells // 2, self.map_metadata.width, step_cells))
            if row_number % 2 == 1:
                column_indices.reverse()

            for col in column_indices:
                cell = self.find_nearest_eligible(row, col, eligible, search_radius)
                if cell is None or cell in seen_cells:
                    continue
                seen_cells.add(cell)
                world = self.cell_to_world(cell[0], cell[1])
                if not waypoints or self.distance(world, waypoints[-1]) > resolution:
                    waypoints.append(world)

        if not waypoints:
            raise RuntimeError("Coverage explorer could not find any reachable waypoints in the occupancy map.")
        return waypoints

    def compute_clearance_mask(self, clearance_cells):
        free = np.array(self.free_mask, dtype=bool)
        if clearance_cells <= 0:
            return free

        integral = np.pad((~free).astype(np.int32), ((1, 0), (1, 0)), mode="constant")
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        eligible = np.zeros_like(free, dtype=bool)

        for row in range(self.map_metadata.height):
            row_min = max(0, row - clearance_cells)
            row_max = min(self.map_metadata.height - 1, row + clearance_cells)
            for col in range(self.map_metadata.width):
                if not free[row, col]:
                    continue
                col_min = max(0, col - clearance_cells)
                col_max = min(self.map_metadata.width - 1, col + clearance_cells)
                blocked = (
                    integral[row_max + 1, col_max + 1]
                    - integral[row_min, col_max + 1]
                    - integral[row_max + 1, col_min]
                    + integral[row_min, col_min]
                )
                eligible[row, col] = blocked == 0
        return eligible

    def find_nearest_eligible(self, row, col, eligible, search_radius):
        if eligible[row, col]:
            return row, col

        best_cell = None
        best_distance = None
        row_min = max(0, row - search_radius)
        row_max = min(self.map_metadata.height - 1, row + search_radius)
        col_min = max(0, col - search_radius)
        col_max = min(self.map_metadata.width - 1, col + search_radius)

        for candidate_row in range(row_min, row_max + 1):
            for candidate_col in range(col_min, col_max + 1):
                if not eligible[candidate_row, candidate_col]:
                    continue
                distance = (candidate_row - row) ** 2 + (candidate_col - col) ** 2
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cell = (candidate_row, candidate_col)
        return best_cell

    def cell_to_world(self, row, col):
        x = self.map_metadata.origin_x + (col + 0.5) * self.map_metadata.resolution
        y = self.map_metadata.origin_y + (self.map_metadata.height - row - 0.5) * self.map_metadata.resolution
        return (float(x), float(y))

    @staticmethod
    def distance(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def main():
    rclpy.init()
    node = CoverageExplorerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
