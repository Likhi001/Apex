import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
import math
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from rclpy.duration import Duration

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Movement constants
LEFT_TURN = +1.0
RIGHT_TURN = -1.0
TURN_MIN = 0.0
TURN_MAX = 1.0

# Speed percentages
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_40_PERCENT = SPEED_MAX * 0.4
SPEED_45_PERCENT = SPEED_MAX * 0.45
SPEED_30_PERCENT = SPEED_MAX * 0.3
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_65_PERCENT = SPEED_MAX * 0.65
SPEED_75_PERCENT = SPEED_25_PERCENT * 3

# Thresholds and constants for obstacle detection and ramp handling
THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25
TURN_SCALING_FACTOR = 0.6
OBSTACLE_SAFE_DISTANCE = 2.0
RAMP_COOLDOWN_TIME = 5.0
RAMP_SLOPE_THRESHOLD = 0.1
RAMP_DETECTION_COUNT = 5
SHARP_TURN_THRESHOLD = 0.5

class LineFollower(Node):
    """
    ROS2 Node for following a line (lane) using edge vectors and obstacle detection.

    Subscriptions:
    - /edge_vectors: Receives edge vectors from image processing.
    - /traffic_status: Receives traffic status updates (e.g., stop signs).
    - /scan: Receives lidar scan data for obstacle detection.

    Publications:
    - /cerebri/in/joy: Publishes joystick-like commands for rover control.
    """

    def __init__(self):
        super().__init__('line_follower')

        try:
            # Subscriptions to edge vectors, traffic status, and lidar scan
            self.subscription_vectors = self.create_subscription(
                EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

            self.publisher_joy = self.create_publisher(
                Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

            self.subscription_traffic = self.create_subscription(
                TrafficStatus, '/traffic_status', self.traffic_status_callback, QOS_PROFILE_DEFAULT)

            self.subscription_lidar = self.create_subscription(
                LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)

            # Initialize variables for obstacle detection and ramp handling
            self.traffic_status = TrafficStatus()
            self.obstacle_detected = False
            self.ramp_detected = False
            self.closest_obstacle_distance = float('inf')
            self.ramp_climbed_time = None
            self.on_ramp = False

        except Exception as e:
            self.get_logger().error(f'Failed to initialize subscriptions or publishers: {str(e)}')

    def rover_move_manual_mode(self, speed, turn):
        """
        Sends joystick-like commands to control the rover's movement.

        Args:
            speed: Speed value for forward/backward motion.
            turn: Turn value for left/right steering.
        """
        try:
            msg = Joy()
            msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, speed, 0.0, turn]
            self.publisher_joy.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish Joy message: {str(e)}')

    def calculate_speed_and_turn(self, deviation, width, sharp_turn):
        """
        Calculates the speed and turn values based on deviation from the lane center.

        Args:
            deviation: The deviation of the rover from the center of the lane.
            width: The width of the image (lane).
            sharp_turn: Boolean indicating if a sharp turn is required.

        Returns:
            Tuple of (speed, turn) values.
        """
        if sharp_turn:
            speed = SPEED_50_PERCENT
            turn = (deviation / width) * TURN_SCALING_FACTOR * 2
        else:
            speed = SPEED_65_PERCENT
            turn = (deviation / width) * TURN_SCALING_FACTOR
        return speed, turn

    def edge_vectors_callback(self, message):
        """
        Callback function to process edge vectors and control rover movement.

        Args:
            message: EdgeVectors message containing the lane edge vectors.
        """
        try:
            speed = SPEED_MAX
            turn = TURN_MIN
            vectors = message
            half_width = vectors.image_width / 2
            sharp_turn = False
            flag = 0

            if vectors.vector_count == 0:
                pass
            elif vectors.vector_count == 1:
                # Single edge vector detected
                deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                sharp_turn = abs(deviation) > SHARP_TURN_THRESHOLD * vectors.image_width
                speed, turn = self.calculate_speed_and_turn(deviation, vectors.image_width, sharp_turn)
            elif vectors.vector_count == 2:
                # Two edge vectors detected, calculate middle point deviation
                middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
                middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2
                middle_x = (middle_x_left + middle_x_right) / 2
                deviation = half_width - middle_x
                sharp_turn = abs(deviation) > SHARP_TURN_THRESHOLD * half_width
                speed, turn = self.calculate_speed_and_turn(deviation, half_width, sharp_turn)

            # Stop sign detection
            if self.traffic_status.stop_sign:
                speed = SPEED_MIN
                self.get_logger().info("Stop sign detected")

            # Ramp handling
            if self.on_ramp:
                speed = SPEED_25_PERCENT
                self.get_logger().info("Slowing down after climbing ramp")
                if self.get_clock().now() - self.ramp_climbed_time > Duration(seconds=RAMP_COOLDOWN_TIME):
                    self.on_ramp = False

            if self.ramp_detected:
                speed = SPEED_45_PERCENT
                self.ramp_climbed_time = self.get_clock().now()
                self.ramp_detected = False
                self.on_ramp = True
                self.get_logger().info("Ramp/bridge detected")
                flag = 1

            # Obstacle avoidance
            if flag == 0 and self.obstacle_detected:
                speed = SPEED_45_PERCENT
                distance = self.closest_obstacle_distance
                if distance < THRESHOLD_OBSTACLE_VERTICAL:
                    turn = RIGHT_TURN * (1 - distance / THRESHOLD_OBSTACLE_VERTICAL) * TURN_SCALING_FACTOR
                else:
                    turn = LEFT_TURN * (distance / THRESHOLD_OBSTACLE_VERTICAL) * TURN_SCALING_FACTOR
                self.get_logger().info("Obstacle detected")

            # Send movement commands to rover
            self.rover_move_manual_mode(speed, turn)

        except Exception as e:
            self.get_logger().error(f'Error in edge_vectors_callback: {str(e)}')

    def traffic_status_callback(self, message):
        """
        Callback function to update the traffic status.

        Args:
            message: TrafficStatus message containing traffic sign data.
        """
        try:
            self.traffic_status = message
        except Exception as e:
            self.get_logger().error(f'Failed to update traffic status: {str(e)}')

    def lidar_callback(self, message):
        """
        Callback function to process lidar scan data for obstacle detection and ramp handling.

        Args:
            message: LaserScan message containing lidar data.
        """
        try:
            self.ramp_detected = False
            self.obstacle_detected = False
            self.closest_obstacle_distance = float('inf')

            ranges = message.ranges
            length = len(ranges)
            ramp_slope_count = 0

            # Detect ramps based on slope between consecutive range measurements
            for i in range(length - 1):
                if ranges[i] < float('inf') and ranges[i + 1] < float('inf'):
                    slope = (ranges[i + 1] - ranges[i]) / message.angle_increment
                    if slope > RAMP_SLOPE_THRESHOLD:
                        ramp_slope_count += 1
                        if ramp_slope_count >= RAMP_DETECTION_COUNT:
                            self.ramp_detected = True
                            break
                    else:
                        ramp_slope_count = 0

            shield_vertical = 4
            shield_horizontal = 1
            theta = math.atan(shield_vertical / shield_horizontal)

            # Focus on the relevant part of the ranges
            ranges = message.ranges[int(length / 4): int(3 * length / 4)]

            length = float(len(ranges))
            front_ranges = ranges[int(length * theta / PI): int(length * (PI - theta) / PI)]
            side_ranges_right = ranges[0: int(length * theta / PI)]
            side_ranges_left = ranges[int(length * (PI - theta) / PI):]

            # Obstacle detection in front
            for i in range(len(front_ranges)):
                if front_ranges[i] < THRESHOLD_OBSTACLE_VERTICAL:
                    self.obstacle_detected = True
                    self.closest_obstacle_distance = min(self.closest_obstacle_distance, front_ranges[i])
                    return

            # Obstacle detection on the sides (left and right)
            for i in range(len(side_ranges_left) - 1, -1, -1):
                if side_ranges_left[i] < THRESHOLD_OBSTACLE_HORIZONTAL:
                    self.obstacle_detected = True
                    self.closest_obstacle_distance = min(self.closest_obstacle_distance, side_ranges_left[i])
                    return

            for i in range(len(side_ranges_right)):
                if side_ranges_right[i] < THRESHOLD_OBSTACLE_HORIZONTAL:
                    self.obstacle_detected = True
                    self.closest_obstacle_distance = min(self.closest_obstacle_distance, side_ranges_right[i])
                    return

            self.obstacle_detected = False

        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        # Initialize and run the LineFollower node
        line_follower = LineFollower()
        rclpy.spin(line_follower)
    except Exception as e:
        rclpy.logging.get_logger('LineFollower').error(f'Unexpected error: {str(e)}')
    finally:
        line_follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
