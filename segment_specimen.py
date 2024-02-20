import cv2
import numpy as np

def segment_specimen(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Convert to HSV and threshold to get the specimen
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # These thresholds might need adjustment
    lower_white = np.array([0, 0, 205])
    upper_white = np.array([172, 151, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Optional: Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find contours and the largest one will be the specimen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    return mask, largest_contour

# Example usage:
image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'
mask, specimen_contour = segment_specimen(image_path)

# To visualize the mask
cv2.imshow('Specimen Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


def calculate_radii_from_contour(contour, image_shape):
    # Assuming contour is a numpy array of shape (n, 1, 2) where n is the number of points
    # and image_shape is the shape of the numpy array of the image (height, width)
    
    # Create an empty image to draw the contour filled
    contour_image = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_image, [contour], -1, color=1, thickness=cv2.FILLED)
    
    # Calculate the centerline (simplified as the vertical center for this example)
    center_x = image_shape[1] // 2
    
    # Initialize an empty list to hold the radii
    radii = []
    
    # For each row in the image, find the first and last white pixel
    for y in range(image_shape[0]):
        row = contour_image[y, :]
        if np.any(row):
            left_most = np.where(row == 1)[0][0]
            right_most = np.where(row == 1)[0][-1]
            radius = (right_most - left_most) / 2
            radii.append((y, center_x - left_most, radius))
    
    return radii

def radii_to_3d_coordinates(radii, angle_degrees):
    points_3d = []
    angle_radians = np.radians(angle_degrees)
    for y, radius in radii:
        x = radius * np.cos(angle_radians)
        z = radius * np.sin(angle_radians)
        points_3d.append((x, y, z))
    return points_3d

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3d_points(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in points_3d:
        ax.scatter(x, y, z)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()
