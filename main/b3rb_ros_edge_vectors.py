import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Colors for drawing vectors in the debug image
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# Constants for vector processing
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.40
VECTOR_MAGNITUDE_MINIMUM = 2.5

class EdgeVectorsPublisher(Node):
    """
    ROS2 Node for processing camera images to detect edge vectors of lanes
    and publishing the vectors for lane following.

    Subscriptions:
    - /camera/image_raw/compressed: Receives compressed images from the camera.

    Publications:
    - /edge_vectors: Publishes detected edge vectors for lane following.
    - /debug_images/thresh_image: Publishes the thresholded binary image for debugging.
    - /debug_images/vector_image: Publishes the image with detected vectors for debugging.
    """

    def __init__(self):
        super().__init__('edge_vectors_publisher')
        try:
            # Subscription to camera image feed
            self.subscription_camera = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                self.camera_image_callback,
                QOS_PROFILE_DEFAULT)

            # Publisher for edge vectors
            self.publisher_edge_vectors = self.create_publisher(
                EdgeVectors,
                '/edge_vectors',
                QOS_PROFILE_DEFAULT)

            # Publishers for debugging images
            self.publisher_thresh_image = self.create_publisher(
                CompressedImage,
                "/debug_images/thresh_image",
                QOS_PROFILE_DEFAULT)

            self.publisher_vector_image = self.create_publisher(
                CompressedImage,
                "/debug_images/vector_image",
                QOS_PROFILE_DEFAULT)

            self.image_height = 0
            self.image_width = 0
            self.lower_image_height = 0
            self.upper_image_height = 0

        except Exception as e:
            self.get_logger().error(f'Failed to initialize subscriptions or publishers: {str(e)}')

    def publish_debug_image(self, publisher, image):
        """
        Helper function to publish debug images.

        Args:
            publisher: The ROS publisher for the debug image.
            image: The image to be published.
        """
        try:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)
        except Exception as e:
            self.get_logger().error(f'Failed to publish debug image: {str(e)}')

    def get_vector_angle_in_radians(self, vector):
        """
        Calculates the angle of a vector in radians.

        Args:
            vector: A pair of points representing the vector.

        Returns:
            The angle of the vector in radians.
        """
        try:
            if (vector[0][0] - vector[1][0]) == 0:
                theta = PI / 2
            else:
                slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
                theta = math.atan(slope)
            return theta
        except Exception as e:
            self.get_logger().error(f'Failed to compute vector angle: {str(e)}')
            return 0.0

    def compute_vectors_from_image(self, image, thresh):
        """
        Processes the thresholded image to find contours and compute edge vectors.

        Args:
            image: The original image (used for drawing).
            thresh: The thresholded binary image.

        Returns:
            A list of vectors and the image with drawn vectors.
        """
        try:
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            vectors = []
            for i in range(len(contours)):
                coordinates = contours[i][:, 0, :]

                min_y_value = np.min(coordinates[:, 1])
                max_y_value = np.max(coordinates[:, 1])

                min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
                max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

                min_y_coord = min_y_coords[0]
                max_y_coord = max_y_coords[0]

                magnitude = np.linalg.norm(min_y_coord - max_y_coord)
                if magnitude > VECTOR_MAGNITUDE_MINIMUM:
                    rover_point = [self.image_width / 2, self.lower_image_height]
                    middle_point = (min_y_coord + max_y_coord) / 2
                    distance = np.linalg.norm(middle_point - rover_point)

                    angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                    if angle > 0:
                        min_y_coord[0] = np.max(min_y_coords[:, 0])
                    else:
                        max_y_coord[0] = np.max(max_y_coords[:, 0])

                    vectors.append([list(min_y_coord), list(max_y_coord)])
                    vectors[-1].append(distance)

                cv2.line(image, min_y_coord, max_y_coord, BLUE_COLOR, 2)

            return vectors, image
        except Exception as e:
            self.get_logger().error(f'Error in compute_vectors_from_image: {str(e)}')
            return [], image

    def process_image_for_edge_vectors(self, image):
        """
        Main function to process the image, detect edges, and compute lane vectors.

        Args:
            image: The input image from the camera.

        Returns:
            A list of final vectors representing the lane edges.
        """
        try:
            self.image_height, self.image_width, _ = image.shape
            self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
            self.upper_image_height = self.image_height - self.lower_image_height

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold_black = 25
            thresh = cv2.threshold(gray, threshold_black, 255, cv2.THRESH_BINARY_INV)[1]

            thresh = thresh[self.image_height - self.lower_image_height:]
            image = image[self.image_height - self.lower_image_height:]
            vectors, image = self.compute_vectors_from_image(image, thresh)

            vectors = sorted(vectors, key=lambda x: x[2])

            half_width = self.image_width / 2
            vectors_left = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) < half_width]
            vectors_right = [i for i in vectors if ((i[0][0] + i[1][0]) / 2) >= half_width]

            final_vectors = []
            for vectors_inst in [vectors_left, vectors_right]:
                if len(vectors_inst) > 0:
                    cv2.line(image, vectors_inst[0][0], vectors_inst[0][1], GREEN_COLOR, 2)
                    vectors_inst[0][0][1] += self.upper_image_height
                    vectors_inst[0][1][1] += self.upper_image_height
                    final_vectors.append(vectors_inst[0][:2])

            self.publish_debug_image(self.publisher_thresh_image, thresh)
            self.publish_debug_image(self.publisher_vector_image, image)

            return final_vectors
        except Exception as e:
            self.get_logger().error(f'Error in process_image_for_edge_vectors: {str(e)}')
            return []

    def camera_image_callback(self, message):
        """
        Callback function to handle incoming camera images and publish edge vectors.

        Args:
            message: CompressedImage message containing the camera image.
        """
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            vectors = self.process_image_for_edge_vectors(image)

            vectors_message = EdgeVectors()
            vectors_message.image_height = image.shape[0]
            vectors_message.image_width = image.shape[1]
            vectors_message.vector_count = 0
            if len(vectors) > 0:
                vectors_message.vector_1[0].x = float(vectors[0][0][0])
                vectors_message.vector_1[0].y = float(vectors[0][0][1])
                vectors_message.vector_1[1].x = float(vectors[0][1][0])
                vectors_message.vector_1[1].y = float(vectors[0][1][1])
                vectors_message.vector_count += 1
            if len(vectors) > 1:
                vectors_message.vector_2[0].x = float(vectors[1][0][0])
                vectors_message.vector_2[0].y = float(vectors[1][0][1])
                vectors_message.vector_2[1].x = float(vectors[1][1][0])
                vectors_message.vector_2[1].y = float(vectors[1][1][1])
                vectors_message.vector_count += 1
            self.publisher_edge_vectors.publish(vectors_message)
        except Exception as e:
            self.get_logger().error(f'Error in camera_image_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        # Initialize and run the EdgeVectorsPublisher node
        edge_vectors_publisher = EdgeVectorsPublisher()
        rclpy.spin(edge_vectors_publisher)
    except Exception as e:
        rclpy.logging.get_logger('EdgeVectorsPublisher').error(f'Unexpected error: {str(e)}')
    finally:
        edge_vectors_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
