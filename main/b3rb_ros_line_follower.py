import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import math
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from sensor_msgs.msg import LaserScan
from rclpy.duration import Duration

QOS_PROFILE_DEFAULT = 10
PI = math.pi
LEFT_TURN = +1.0
RIGHT_TURN = -1.0
TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3
THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25
TURN_SCALING_FACTOR = 0.5  # Scale down the turn value to make the turn more gradual
OBSTACLE_SAFE_DISTANCE = 2.0  # Safe distance from obstacle in meters
RAMP_COOLDOWN_TIME = 5.0  # Time in seconds to slow down after climbing a ramp

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Subscription for edge vectors.
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for joy (for moving the rover in manual mode).
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Subscription for traffic status.
        self.subscription_traffic = self.create_subscription(
            TrafficStatus,
            '/traffic_status',
            self.traffic_status_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for LIDAR data.
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.traffic_status = TrafficStatus()
        self.obstacle_detected = False
        self.ramp_detected = False
        self.closest_obstacle_distance = float('inf')
        self.ramp_climbed_time = None

    def rover_move_manual_mode(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    def calculate_speed_and_turn(self, distance):
        if distance > OBSTACLE_SAFE_DISTANCE:
            speed = SPEED_MAX
            turn = TURN_MIN
        else:
            speed = max(SPEED_MIN, SPEED_MAX * (distance / OBSTACLE_SAFE_DISTANCE))
            turn = max(TURN_MIN, TURN_MAX * (1 - (distance / OBSTACLE_SAFE_DISTANCE)))
        return speed, turn

    def edge_vectors_callback(self, message):
        speed = SPEED_MAX
        turn = TURN_MIN

        vectors = message
        half_width = vectors.image_width / 2

        if vectors.vector_count == 0:
            pass

        elif vectors.vector_count == 1:
            deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
            turn = (deviation / vectors.image_width) * TURN_SCALING_FACTOR
            # Adjust speed for sharper turns
            if abs(turn) > 0.5:
                speed = SPEED_50_PERCENT

        elif vectors.vector_count == 2:
            middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
            middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2
            middle_x = (middle_x_left + middle_x_right) / 2
            deviation = half_width - middle_x
            turn = (deviation / half_width) * TURN_SCALING_FACTOR
            # Adjust speed for sharper turns
            if abs(turn) > 0.5:
                speed = SPEED_50_PERCENT

        if self.ramp_detected:
            speed = SPEED_25_PERCENT
            self.ramp_climbed_time = self.get_clock().now()
            self.ramp_detected = False
            print("Ramp/bridge detected")

        if self.ramp_climbed_time is not None:
            if self.get_clock().now() - self.ramp_climbed_time < Duration(seconds=RAMP_COOLDOWN_TIME):
                speed = SPEED_50_PERCENT
                print("Slowing down after climbing ramp")

        if self.obstacle_detected:
            speed = SPEED_25_PERCENT
            if vectors.vector_count == 0:
                turn = TURN_MIN
                pass
            elif vectors.vector_count == 1:
                if vectors.vector_1[1].x < half_width:
                    turn = LEFT_TURN * TURN_SCALING_FACTOR
                else:
                    turn = RIGHT_TURN * TURN_SCALING_FACTOR
            else:
                speed = SPEED_MAX
                turn = TURN_MIN
            print("Obstacle detected")

        self.rover_move_manual_mode(speed, turn)

    def traffic_status_callback(self, message):
        self.traffic_status = message

    def lidar_callback(self, message):
        self.ramp_detected = False
        self.obstacle_detected = False

        shield_vertical = 4
        shield_horizontal = 1
        theta = math.atan(shield_vertical / shield_horizontal)

        length = float(len(message.ranges))
        ranges = message.ranges[int(length / 4): int(3 * length / 4)]

        length = float(len(ranges))
        front_ranges = ranges[int(length * theta / PI): int(length * (PI - theta) / PI)]
        side_ranges_right = ranges[0: int(length * theta / PI)]
        side_ranges_left = ranges[int(length * (PI - theta) / PI):]

        angle = theta - PI / 2
        for i in range(len(front_ranges)):
            if front_ranges[i] < THRESHOLD_OBSTACLE_VERTICAL:
                self.obstacle_detected = True
                return
            angle += message.angle_increment

        side_ranges_left.reverse()
        for side_ranges in [side_ranges_left, side_ranges_right]:
            angle = 0.0
            for i in range(len(side_ranges)):
                if side_ranges[i] < THRESHOLD_OBSTACLE_HORIZONTAL:
                    self.obstacle_detected = True
                    return
                angle += message.angle_increment

        self.obstacle_detected = False

def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()
    rclpy.spin(line_follower)
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()