import cv2
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor

# Load the scaling factor
with open("scaling_factor.txt", "r") as f:
    scaling_factor = float(f.read())

# Define the HSV range for the metal cylinder (specimen)
metal_cylinder_lower = np.array([5, 20, 50], dtype=np.uint8)
metal_cylinder_upper = np.array([15, 255, 240], dtype=np.uint8)

# Define the HSV range for the blue tape
blue_tape_lower = np.array([100, 150, 50], dtype=np.uint8)
blue_tape_upper = np.array([130, 255, 255], dtype=np.uint8)

def smooth_radii(radii, window_size=15, polyorder=2):
    smoothed_radii = savgol_filter(radii, window_size, polyorder)
    return smoothed_radii

#To isolate the metal cylinder and find its edges
def segment_specimen(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply the color range masks
    mask_metal_cylinder = cv2.inRange(hsv_image, metal_cylinder_lower, metal_cylinder_upper)
    mask_blue_tape = cv2.inRange(hsv_image, blue_tape_lower, blue_tape_upper)
    
    # Remove the blue tape from the specimen mask
    specimen_only_mask = cv2.bitwise_and(mask_metal_cylinder, cv2.bitwise_not(mask_blue_tape))
    
    # Apply the mask to the grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    specimen_only_image = cv2.bitwise_and(gray_image, gray_image, mask=specimen_only_mask)
    
    # Use Canny edge detection on the specimen-only image
    edges = cv2.Canny(specimen_only_image, 50, 150)
    
    return edges, mask_blue_tape, image

#To calculate the centerline of the metal cylinder and determine the radii at different heights and angles, using the blue tape as a reference
def calculate_centerline_and_radii(edges, mask_blue_tape, scaling_factor):
    contours, _ = cv2.findContours(mask_blue_tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        cx = int(M["m10"] / M["m00"])

        radii = []
        for y in range(edges.shape[0]):
            row = edges[y, :]
            white_pixels = np.where(row > 0)[0]
            if white_pixels.size > 0:
                radius_left = cx - white_pixels[0]
                radius_right = white_pixels[-1] - cx
                radius = max(radius_left, radius_right)  # Use max to avoid extending to the center
                radius_mm = radius * scaling_factor
                height = y * scaling_factor
                radii.append((height, radius_mm))

        # Using RANSAC for robust outlier detection
        heights, radii_values = zip(*radii)
        ransac = RANSACRegressor()
        ransac.fit(np.array(heights).reshape(-1, 1), radii_values)
        inlier_mask = ransac.inlier_mask_

        radii = [(heights[i], radii_values[i]) for i in range(len(heights)) if inlier_mask[i]]

        # Smooth the radii
        heights, radii_values = zip(*radii)
        smoothed_radii_values = smooth_radii(radii_values)

        smoothed_radii = list(zip(heights, smoothed_radii_values))

        return smoothed_radii, cx * scaling_factor
    else:
        return [], edges.shape[1] // 2 * scaling_factor

#To extract radii data from each image in a folder
def process_image(image_path, angle, scaling_factor):
    edges, mask_blue_tape, _ = segment_specimen(image_path)
    radii, center_x = calculate_centerline_and_radii(edges, mask_blue_tape, scaling_factor)
    return angle, radii, center_x

def process_folder(folder_path, scaling_factor):
    angles_and_radii = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".JPG")])
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        angle = idx * 360 / len(image_files)
        angle, radii, center_x = process_image(image_path, angle, scaling_factor)
        angles_and_radii.append((angle, radii, center_x))
    return angles_and_radii

#To visualize the measured radii in a 3D space and compare them to an ideal cylinder
def plot_3d_radii(angles_and_radii):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for angle_deg, radii, center_x in angles_and_radii:
        angle_rad = np.radians(angle_deg)
        x_points = []
        y_points = []
        z_points = []

        if radii:
            max_height = max(height for height, _ in radii)
            scale_factor = 4 / max_height
            for height, radius in radii:
                height = height * scale_factor  # Adjust height to ensure 4mm maximum

                x = radius * np.cos(angle_rad)
                y = radius * np.sin(angle_rad)
                z = height
                x_points.append(x)
                y_points.append(y)
                z_points.append(z)
            
            ax.plot(x_points, y_points, z_points, color='r')

    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, 4, 100)
    theta, z = np.meshgrid(theta, z)
    x = 4 * np.cos(theta)
    y = 4 * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=0.3, color='b')

    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=5, label='Measured Points')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=5, label='Ideal Cylinder')
    ax.legend(handles=[red_patch, blue_patch])

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Height (mm)')
    plt.title('3D Radii Plot')
    plt.show()

#To show the average radius at different heights along the cylinder and compare it to the ideal radius (4mm)
def plot_radii_vs_height(angles_and_radii):
    radii_dict = collections.defaultdict(list)

    for _, radii_list, _ in angles_and_radii:
        for height, radius in radii_list:
            radii_dict[height].append(radius)

    heights = []
    avg_radii = []
    for height, radii in sorted(radii_dict.items()):
        avg_radius = sum(radii) / len(radii)
        heights.append(height)
        avg_radii.append(avg_radius)

    max_height = max(heights)
    scale_factor = 4 / max_height
    heights = [height * scale_factor for height in heights]

    avg_radii_array = np.array(avg_radii)
    heights_array = np.array(heights)

    ideal_radii = np.full_like(avg_radii_array, 4)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(heights_array, avg_radii_array, color='r', label='Measured Radius', linestyle='-', marker='o')
    ax.plot(heights_array, ideal_radii, color='b', linestyle='--', label='Ideal Radius')

    ax.set_xlabel('Height (mm)')
    ax.set_ylabel('Radius (mm)')
    ax.set_title('Average Radius vs Height')
    ax.legend()

    plt.show()

    errors = np.abs(avg_radii_array - ideal_radii)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"Mean error: {mean_error:.3f} mm")
    print(f"Max error: {max_error:.3f} mm")

#To calculate and visualize the error between measured radii and the ideal cylinder
def validate_with_ideal_cylinder(angles_and_radii):
    ideal_radius = 4  # mm
    max_height = 4  # mm

    errors = []
    heights = []

    for _, radii, _ in angles_and_radii:
        for height, radius in radii:
            if height <= max_height:
                ideal_radius_at_height = ideal_radius
                error = euclidean([ideal_radius_at_height], [radius])
                errors.append(error)
                heights.append(height)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"Mean error compared to ideal cylinder: {mean_error:.3f} mm")
    print(f"Max error compared to ideal cylinder: {max_error:.3f} mm")

    # Plot errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(heights, errors, color='r', label='Error', linestyle='-', marker='o')
    ax.set_xlabel('Height (mm)')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Error Distribution')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    folder_path = "/Users/vamsikrishna/Desktop/NewTestImages/"
    angles_and_radii = process_folder(folder_path, scaling_factor)
    plot_3d_radii(angles_and_radii)
    plot_radii_vs_height(angles_and_radii)
    validate_with_ideal_cylinder(angles_and_radii)
