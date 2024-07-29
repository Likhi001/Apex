# b3rb_ros_edge_vectors.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

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

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

    def publish_debug_image(self, publisher, image):
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        if (vector[0][0] - vector[1][0]) == 0:
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def compute_vectors_from_image(self, image, thresh):
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

    def process_image_for_edge_vectors(self, image):
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

    def camera_image_callback(self, message):
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

def main(args=None):
    rclpy.init(args=args)
    edge_vectors_publisher = EdgeVectorsPublisher()
    rclpy.spin(edge_vectors_publisher)
    edge_vectors_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
