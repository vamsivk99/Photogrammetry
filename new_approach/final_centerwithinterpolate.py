import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

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
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:  # To avoid division by zero
            center_x = int(M["m10"] / M["m00"])  # Calculate the x-coordinate of the center of mass
        else:
            center_x = 0
        radii = []

        for point in largest_contour:
            x, y = point[0]
            distance = np.sqrt((x - center_x) ** 2)
            radii.append((y, distance))

        # Draw the centerline
        cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 2)

        return sorted(radii, key=lambda x: x[0]), center_x, image  # Return the image with the centerline
    return [], image.shape[1] // 2, image  # Return the original image if no contours found

if __name__ == "__main__":
    folder_path = '/Users/vamsikrishna/Desktop/TestImages/'  # Replace with your folder path

    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG"):  # Check if the file is an image
            image_path = os.path.join(folder_path, filename)
            result, mask_blue, _ = segment_specimen(image_path)
            _, _, image_with_centerline = calculate_centerline_and_radii(result, mask_blue)

            # Display the image with the centerline
            cv2.imshow('Image with Centerline', image_with_centerline)
            cv2.waitKey(500)  # Wait for 500 ms

def plot_3d_radii_delaunay(angles_and_radii):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Collect all points
    points = []
    for angle, rad_height_pairs in angles_and_radii:
        angle_rad = np.radians(angle)
        for height, radius in rad_height_pairs:
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            z = height
            points.append([x, y, z])
            
    points = np.array(points)
    if points.shape[0] < 3:  # Check if enough points to triangulate
        print("Not enough points to generate a 3D plot.")
        return

    # Perform Delaunay triangulation
    tri = Delaunay(points[:, :2])  # Triangulation on x, y
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    plt.show()

def process_folder(folder_path):
    angles_and_radii = []
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".JPG"):
            image_path = os.path.join(folder_path, filename)
            angle = idx * 360 / len(os.listdir(folder_path))
            image, mask_blue, _ = segment_specimen(image_path)
            radii, center_x, _ = calculate_centerline_and_radii(image, mask_blue)  # Update this line
            angles_and_radii.append((angle, radii))

    if angles_and_radii:
        plot_3d_radii_delaunay(angles_and_radii)
    else:
        print("No data to plot.")

if __name__ == "__main__":
    folder_path = "/Users/vamsikrishna/Desktop/TestImages/"
    process_folder(folder_path)