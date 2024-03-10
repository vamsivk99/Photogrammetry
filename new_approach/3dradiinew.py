import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def segment_specimen(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for colors we want to exclude (blue tape and orange background)
    orange_lower = np.array([5, 20, 50], dtype=np.uint8)
    orange_upper = np.array([15, 255, 240], dtype=np.uint8)
    blue_lower = np.array([35, 0, 0], dtype=np.uint8)
    blue_upper = np.array([120, 255, 255], dtype=np.uint8)
    
    # Create masks and remove these colors
    mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_remove = cv2.bitwise_or(mask_orange, mask_blue)
    mask_specimen = cv2.bitwise_not(mask_remove)
    
    # Apply the mask to highlight the specimen and the blue tape for centerline calculation
    result = cv2.bitwise_and(image, image, mask=mask_specimen)
    return result, mask_blue, image

def calculate_centerline_and_radii(image, mask_blue):
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour assumed to be the blue tape
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        cx = int(M["m10"] / M["m00"])  # Center x of blue tape

        # Calculate radii from centerline to specimen edge at various heights
        radii = []
        for y in range(image.shape[0]):
            row = image[y, :, 0]  # Using the first channel as reference
            # Find the first and last white pixel in the row (specimen edge)
            white_pixels = np.where(row > 0)[0]
            if white_pixels.size > 0:
                radius_left = cx - white_pixels[0]
                radius_right = white_pixels[-1] - cx
                radius = max(radius_left, radius_right)  # Max to get the furthest edge
                radii.append((y, radius))
        return radii, cx
    else:
        return [], image.shape[1] // 2  # Default centerline if no blue tape found

def process_image(image_path, angle):
    image, mask_blue, _ = segment_specimen(image_path)
    radii, center_x = calculate_centerline_and_radii(image, mask_blue)
    return angle, radii, center_x

def process_folder(folder_path):
    angles_and_radii = []  # Store (angle, radii, center_x) tuples
    # Angle calculation logic based on image ordering or filenames
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".JPG"):
            image_path = os.path.join(folder_path, filename)
            angle = idx * 360 / len(os.listdir(folder_path))  # Example angle calculation
            print(f"Processing {image_path} at angle {angle}")
            angle, radii, center_x = process_image(image_path, angle)
            angles_and_radii.append((angle, radii, center_x))
    return angles_and_radii


# def plot_3d_radii(angles_and_radii):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for angle, radii, _ in angles_and_radii:  # Unpack three items instead of two
#         angle_rad = np.radians(angle)
#         for height, radius in radii:
#             x = radius * np.cos(angle_rad)
#             y = radius * np.sin(angle_rad)
#             z = height
#             ax.scatter(x, y, z, c='r', marker='o')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Height')

#     for angle in range(0, 360, 10):  # Rotate the plot by 10 degrees at a time
#         ax.view_init(30, angle)  # 30 is the elevation angle, change as needed
#         plt.draw()
#         plt.pause(0.1)  # Pause to allow the plot to update

#     plt.show()

def plot_3d_radii(angles_and_radii):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a list for x, y, z coordinates for all points
    x_points = []
    y_points = []
    z_points = []

    # Iterate over angles and radii and convert polar coordinates to Cartesian
    for angle_deg, radii, center_x in angles_and_radii:  # Adjusted to unpack three items
        angle_rad = np.radians(angle_deg)
        for height, radius in radii:
            x = (radius + center_x) * np.cos(angle_rad)  # Adjust x based on center_x
            y = (radius + center_x) * np.sin(angle_rad)  # Adjust y based on center_x
            z = height
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)

    # Convert lists to numpy arrays for plotting
    x_array = np.array(x_points)
    y_array = np.array(y_points)
    z_array = np.array(z_points)

    # Plot the points
    ax.scatter(x_array, y_array, z_array, c='r', marker='o')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')

    plt.show()

# Test data - you should replace this with your actual data
angles_and_radii = [(i, [(h, np.sin(np.radians(i)) * 100 + 100) for h in range(0, 300)], 100) for i in range(0, 360, 10)]

plot_3d_radii(angles_and_radii)


if __name__ == "__main__":
    folder_path = "/Users/vamsikrishna/Desktop/TestImages/"
    angles_and_radii = process_folder(folder_path)
    plot_3d_radii(angles_and_radii)


# def plot_3d_radii(angles_and_radii):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Create a list for x, y, z coordinates for all points
#     x_points = []
#     y_points = []
#     z_points = []

#     # Iterate over angles and radii and convert polar coordinates to Cartesian
#     for angle_deg, radii, center_x in angles_and_radii:  # Adjusted to unpack three items
#         angle_rad = np.radians(angle_deg)
#         for height, radius in radii:
#             x = (radius + center_x) * np.cos(angle_rad)  # Adjust x based on center_x
#             y = (radius + center_x) * np.sin(angle_rad)  # Adjust y based on center_x
#             z = height
#             x_points.append(x)
#             y_points.append(y)
#             z_points.append(z)

#     # Convert lists to numpy arrays for plotting
#     x_array = np.array(x_points)
#     y_array = np.array(y_points)
#     z_array = np.array(z_points)

#     # Plot the points
#     ax.scatter(x_array, y_array, z_array, c='r', marker='o')

#     # Label the axes
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Height')

#     plt.show()

# # Test data - you should replace this with your actual data
# angles_and_radii = [(i, [(h, np.sin(np.radians(i)) * 100 + 100) for h in range(0, 300)], 100) for i in range(0, 360, 10)]

# plot_3d_radii(angles_and_radii)


# if __name__ == "__main__":
#     folder_path = "/Users/vamsikrishna/Desktop/TestImages/"
#     angles_and_radii = process_folder(folder_path)
#     plot_3d_radii(angles_and_radii)
