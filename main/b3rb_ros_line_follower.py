import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
import math
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from rclpy.duration import Duration

# Constants for QoS, directions, and speeds
QOS_PROFILE_DEFAULT = 10
PI = math.pi
LEFT_TURN, RIGHT_TURN = +1.0, -1.0
TURN_MIN, TURN_MAX = 0.0, 1.0
SPEED_MIN, SPEED_MAX = 0.0, 1.0

# Speed constants as percentages of SPEED_MAX
SPEED_25_PERCENT = SPEED_MAX * 0.25
SPEED_40_PERCENT = SPEED_MAX * 0.4
SPEED_45_PERCENT = SPEED_MAX * 0.45
SPEED_50_PERCENT = SPEED_MAX * 0.5
SPEED_65_PERCENT = SPEED_MAX * 0.65

# Thresholds and factors
THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25
TURN_SCALING_FACTOR = 0.6
RAMP_COOLDOWN_TIME = 5.0
RAMP_SLOPE_THRESHOLD = 0.1
RAMP_DETECTION_COUNT = 5
SHARP_TURN_THRESHOLD = 0.5

class LineFollower(Node):
    def __init__(self):
        """Initialize the LineFollower node."""
        super().__init__('line_follower')

        try:
            # Set up subscriptions and publishers
            self.create_subscriptions()
            self.create_publishers()

            # Initialize state variables
            self.traffic_status = TrafficStatus()
            self.obstacle_detected = False
            self.ramp_detected = False
            self.closest_obstacle_distance = float('inf')
            self.ramp_climbed_time = None
            self.on_ramp = False

        except Exception as e:
            self.get_logger().error(f'Initialization error: {str(e)}')

    def create_subscriptions(self):
        """Create all necessary subscriptions."""
        self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(TrafficStatus, '/traffic_status', self.traffic_status_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)

    def create_publishers(self):
        """Create all necessary publishers."""
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

    def rover_move_manual_mode(self, speed, turn):
        """Publish a Joy message to control the rover's movement."""
        try:
            msg = Joy()
            msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, speed, 0.0, turn]
            self.publisher_joy.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish Joy message: {str(e)}')

    def calculate_speed_and_turn(self, deviation, width, sharp_turn):
        """Calculate speed and turn values based on deviation and turn sharpness."""
        if sharp_turn:
            speed = SPEED_50_PERCENT
            turn = (deviation / width) * TURN_SCALING_FACTOR * 2
        else:
            speed = SPEED_65_PERCENT
            turn = (deviation / width) * TURN_SCALING_FACTOR
        return speed, turn

    def edge_vectors_callback(self, message):
        """Process edge vectors to determine rover movement."""
        try:
            speed, turn = self.process_edge_vectors(message)
            speed, turn = self.adjust_for_conditions(speed, turn)
            self.rover_move_manual_mode(speed, turn)
        except Exception as e:
            self.get_logger().error(f'Error in edge_vectors_callback: {str(e)}')

    def process_edge_vectors(self, vectors):
        """Process edge vectors to calculate initial speed and turn."""
        half_width = vectors.image_width / 2
        if vectors.vector_count == 1:
            deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
            sharp_turn = abs(deviation) > SHARP_TURN_THRESHOLD * vectors.image_width
            return self.calculate_speed_and_turn(deviation, vectors.image_width, sharp_turn)
        elif vectors.vector_count == 2:
            middle_x = ((vectors.vector_1[0].x + vectors.vector_1[1].x) / 2 +
                        (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2) / 2
            deviation = half_width - middle_x
            sharp_turn = abs(deviation) > SHARP_TURN_THRESHOLD * half_width
            return self.calculate_speed_and_turn(deviation, half_width, sharp_turn)
        return SPEED_MAX, TURN_MIN

    def adjust_for_conditions(self, speed, turn):
        """Adjust speed and turn based on detected conditions."""
        if self.traffic_status.stop_sign:
            speed = SPEED_MIN
            self.get_logger().info("Stop sign detected")
        elif self.on_ramp:
            speed = SPEED_25_PERCENT
            self.get_logger().info("Slowing down after climbing ramp")
            if self.get_clock().now() - self.ramp_climbed_time > Duration(seconds=RAMP_COOLDOWN_TIME):
                self.on_ramp = False
        elif self.ramp_detected:
            speed = SPEED_45_PERCENT
            self.ramp_climbed_time = self.get_clock().now()
            self.ramp_detected = False
            self.on_ramp = True
            self.get_logger().info("Ramp/bridge detected")
        elif self.obstacle_detected:
            speed = SPEED_45_PERCENT
            distance = self.closest_obstacle_distance
            if distance < THRESHOLD_OBSTACLE_VERTICAL:
                turn = RIGHT_TURN * (1 - distance / THRESHOLD_OBSTACLE_VERTICAL) * TURN_SCALING_FACTOR
            else:
                turn = LEFT_TURN * (distance / THRESHOLD_OBSTACLE_VERTICAL) * TURN_SCALING_FACTOR
            self.get_logger().info("Obstacle detected")
        return speed, turn

    def traffic_status_callback(self, message):
        """Update traffic status."""
        self.traffic_status = message

    def lidar_callback(self, message):
        """Process LIDAR data to detect obstacles and ramps."""
        try:
            self.process_lidar_data(message)
        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {str(e)}')

    def process_lidar_data(self, message):
        """Process LIDAR data to detect obstacles and ramps."""
        self.detect_ramp(message.ranges)
        self.detect_obstacles(message)

    def detect_ramp(self, ranges):
        """Detect ramps using LIDAR data."""
        ramp_slope_count = 0
        for i in range(len(ranges) - 1):
            if ranges[i] < float('inf') and ranges[i + 1] < float('inf'):
                slope = (ranges[i + 1] - ranges[i]) /message.angle_increment
                if slope > RAMP_SLOPE_THRESHOLD:
                    ramp_slope_count += 1
                    if ramp_slope_count >= RAMP_DETECTION_COUNT:
                        self.ramp_detected = True
                        return
                else:
                    ramp_slope_count = 0

    def detect_obstacles(self, message):
        """Detect obstacles using LIDAR data."""
        ranges = message.ranges[len(message.ranges)//4 : 3*len(message.ranges)//4]
        length = float(len(ranges))
        
        shield_vertical = 4
        shield_horizontal = 1
        theta = math.atan(shield_vertical / shield_horizontal)

        front_ranges = ranges[int(length * theta / PI): int(length * (PI - theta) / PI)]
        side_ranges_right = ranges[0: int(length * theta / PI)]
        side_ranges_left = ranges[int(length * (PI - theta) / PI):]

        self.check_ranges(front_ranges, THRESHOLD_OBSTACLE_VERTICAL)
        self.check_ranges(side_ranges_left, THRESHOLD_OBSTACLE_HORIZONTAL)
        self.check_ranges(side_ranges_right, THRESHOLD_OBSTACLE_HORIZONTAL)

    def check_ranges(self, ranges, threshold):
        """Check a set of ranges for obstacles."""
        for range_value in ranges:
            if range_value < threshold:
                self.obstacle_detected = True
                self.closest_obstacle_distance = min(self.closest_obstacle_distance, range_value)
                return
        self.obstacle_detected = False

def main(args=None):
    rclpy.init(args=args)
    try:
        line_follower = LineFollower()
        rclpy.spin(line_follower)
    except Exception as e:
        rclpy.logging.get_logger('LineFollower').error(f'Unexpected error: {str(e)}')
    finally:
        line_follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()