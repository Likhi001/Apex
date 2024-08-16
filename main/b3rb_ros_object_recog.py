import rclpy
from rclpy.node import Node
from synapse_msgs.msg import TrafficStatus
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

QOS_PROFILE_DEFAULT = 10

class ObjectRecognizer(Node):
    """
    ROS2 Node for recognizing objects (traffic signs) from camera images
    and publishing traffic status.

    Subscriptions:
    - /camera/image_raw/compressed: Receives compressed images from the camera.

    Publications:
    - /traffic_status: Publishes detected traffic status information.
    """

    def __init__(self):
        super().__init__('object_recognizer')

        try:
            # Subscription for camera images
            self.subscription_camera = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.camera_image_callback,
                QOS_PROFILE_DEFAULT)

            # Publisher for traffic status
            self.publisher_traffic = self.create_publisher(
                TrafficStatus,
                '/traffic_status',
                QOS_PROFILE_DEFAULT)

        except Exception as e:
            self.get_logger().error(f'Failed to initialize subscriptions or publishers: {str(e)}')

    def camera_image_callback(self, message):
        """
        Callback function to process received camera images for traffic sign detection.

        Args:
            message: CompressedImage message containing the raw image data.
        """
        try:
            # Convert message to an image
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            traffic_status_message = TrafficStatus()

            # NOTE: Add logic for recognizing traffic signs here

            # Publish traffic status based on detected signs
            self.publisher_traffic.publish(traffic_status_message)

        except Exception as e:
            self.get_logger().error(f'Failed to process camera image or publish traffic status: {str(e)}')

def main(args=None):
    rclpy.init(args=args)

    try:
        # Initialize and run the ObjectRecognizer node
        object_recognizer = ObjectRecognizer()
        rclpy.spin(object_recognizer)
    except Exception as e:
        rclpy.logging.get_logger('ObjectRecognizer').error(f'Unexpected error: {str(e)}')
    finally:
        object_recognizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
