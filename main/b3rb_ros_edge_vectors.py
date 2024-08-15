import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

# Constants
QOS_PROFILE_DEFAULT = 10
PI = math.pi
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.40
VECTOR_MAGNITUDE_MINIMUM = 2.5

class EdgeVectorsPublisher(Node):
    def __init__(self):
        super().__init__('edge_vectors_publisher')
        try:
            # Create subscriptions and publishers
            self.subscription_camera = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.camera_image_callback,
                QOS_PROFILE_DEFAULT)

            self.publisher_edge_vectors = self.create_publisher(
                EdgeVectors,
                '/edge_vectors',
                QOS_PROFILE_DEFAULT)

            self.publisher_thresh_image = self.create_publisher(
                CompressedImage,
                "/debug_images/thresh_image",
                QOS_PROFILE_DEFAULT)

            self.publisher_vector_image = self.create_publisher(
                CompressedImage,
                "/debug_images/vector_image",
                QOS_PROFILE_DEFAULT)

            # Initialize image dimensions
            self.image_height = 0
            self.image_width = 0
            self.lower_image_height = 0
            self.upper_image_height = 0

        except Exception as e:
            self.get_logger().error(f'Failed to initialize subscriptions or publishers: {str(e)}')

    def publish_debug_image(self, publisher, image):
        """Publish a debug image to a specified topic"""
        try:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {str(e)}')

    def get_vector_angle_in_radians(self, vector):
        """Calculate the angle of a vector in radians"""
        try:
            dx = vector[0][0] - vector[1][0]
            dy = vector[1][1] - vector[0][1]
            
            if dx == 0:
                return PI / 2 if dy > 0 else -PI / 2
            
            return math.atan2(dy, dx)
        except Exception as e:
            self.get_logger().error(f'Failed to compute vector angle: {str(e)}')
            return 0.0

    def compute_vectors_from_image(self, image, thresh):
        """Compute vectors from the thresholded image"""
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            vectors = []
            
            for contour in contours:
                # Find the top and bottom points of the contour
                min_y_coord = contour[contour[:, :, 1].argmin()][0]
                max_y_coord = contour[contour[:, :, 1].argmax()][0]

                magnitude = np.linalg.norm(min_y_coord - max_y_coord)
                if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                    rover_point = np.array([self.image_width / 2, self.lower_image_height])
                    middle_point = (min_y_coord + max_y_coord) / 2
                    distance = np.linalg.norm(middle_point - rover_point)

                    angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                    
                    # Adjust x-coordinate based on angle
                    if angle > 0:
                        min_y_coord[0] = np.max(contour[:, :, 0])
                    else:
                        max_y_coord[0] = np.max(contour[:, :, 0])

                    vectors.append([list(min_y_coord), list(max_y_coord), distance])
                    cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

            return vectors, image
        except Exception as e:
            self.get_logger().error(f'Error in compute_vectors_from_image: {str(e)}')
            return [], image

    def process_image_for_edge_vectors(self, image):
        """Process the image to find edge vectors"""
        try:
            self.image_height, self.image_width, _ = image.shape
            self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
            self.upper_image_height = self.image_height - self.lower_image_height

            # Convert to grayscale and threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

            # Focus on the lower part of the image
            thresh = thresh[self.upper_image_height:]
            image = image[self.upper_image_height:]
            
            vectors, image = self.compute_vectors_from_image(image, thresh)

            # Sort vectors by distance and separate left and right vectors
            vectors.sort(key=lambda x: x[2])
            half_width = self.image_width / 2
            vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
            vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

            final_vectors = []
            for vectors_side in [vectors_left, vectors_right]:
                if vectors_side:
                    cv2.line(image, tuple(vectors_side[0][0]), tuple(vectors_side[0][1]), GREEN_COLOR, 2)
                    # Adjust y-coordinates back to full image scale
                    vectors_side[0][0][1] += self.upper_image_height
                    vectors_side[0][1][1] += self.upper_image_height
                    final_vectors.append(vectors_side[0][:2])

            self.publish_debug_image(self.publisher_thresh_image, thresh)
            self.publish_debug_image(self.publisher_vector_image, image)

            return final_vectors
        except Exception as e:
            self.get_logger().error(f'Error in process_image_for_edge_vectors: {str(e)}')
            return []

    def camera_image_callback(self, message):
        """Callback function for processing camera images"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            vectors = self.process_image_for_edge_vectors(image)

            # Prepare EdgeVectors message
            vectors_message = EdgeVectors()
            vectors_message.image_height = image.shape[0]
            vectors_message.image_width = image.shape[1]
            vectors_message.vector_count = len(vectors)

            # Populate vector data in the message
            for i, vector in enumerate(vectors[:2], 1):
                setattr(vectors_message, f'vector_{i}', [
                    {'x': float(vector[0][0]), 'y': float(vector[0][1])},
                    {'x': float(vector[1][0]), 'y': float(vector[1][1])}
                ])

            self.publisher_edge_vectors.publish(vectors_message)
        except Exception as e:
            self.get_logger().error(f'Error in camera_image_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        edge_vectors_publisher = EdgeVectorsPublisher()
        rclpy.spin(edge_vectors_publisher)
    except Exception as e:
        rclpy.logging.get_logger('EdgeVectorsPublisher').error(f'Unexpected error: {str(e)}')
    finally:
        edge_vectors_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()