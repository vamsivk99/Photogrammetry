import cv2
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
import csv

# Load the scaling factor from a file
with open("scaling_factor.txt", "r") as f:
    scaling_factor = float(f.read())

# Define the HSV range for the metal cylinder (specimen)
metal_cylinder_lower = np.array([5, 20, 50], dtype=np.uint8)
metal_cylinder_upper = np.array([15, 255, 240], dtype=np.uint8)

# Define the HSV range for the blue tape
blue_tape_lower = np.array([100, 150, 50], dtype=np.uint8)
blue_tape_upper = np.array([130, 255, 255], dtype=np.uint8)

def smooth_radii(radii, window_size=15, polyorder=2):
    """
    Smooth the radii using the Savitzky-Golay filter.
    """
    smoothed_radii = savgol_filter(radii, window_size, polyorder)
    return smoothed_radii

def segment_specimen(image_path):
    """
    Isolate the metal cylinder and find its edges using color segmentation and Canny edge detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open/read file: {image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply the color range masks to isolate the metal cylinder
    mask_metal_cylinder = cv2.inRange(hsv_image, metal_cylinder_lower, metal_cylinder_upper)
    mask_blue_tape = cv2.inRange(hsv_image, blue_tape_lower, blue_tape_upper)
    
    # Remove the blue tape from the specimen mask
    specimen_only_mask = cv2.bitwise_and(mask_metal_cylinder, cv2.bitwise_not(mask_blue_tape))
    
    # Apply the mask to the grayscale image to isolate the specimen
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    specimen_only_image = cv2.bitwise_and(gray_image, gray_image, mask=specimen_only_mask)
    
    # Use Canny edge detection on the isolated specimen
    edges = cv2.Canny(specimen_only_image, 50, 150)
    
    return edges, mask_blue_tape, image

def calculate_centerline_and_radii(edges, mask_blue_tape, scaling_factor):
    """
    Calculate the centerline of the metal cylinder and determine the radii at different heights and angles.
    """
    contours, _ = cv2.findContours(mask_blue_tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        cx = int(M["m10"] / M["m00"])
        print(f"Detected center (cx): {cx}")

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

        return smoothed_radii, cx
    else:
        return [], edges.shape[1] // 2

def process_image(image_path, angle, scaling_factor):
    """
    Extract radii data from an image.
    """
    edges, mask_blue_tape, _ = segment_specimen(image_path)
    radii, center_x = calculate_centerline_and_radii(edges, mask_blue_tape, scaling_factor)
    return angle, radii, center_x

def process_folder(folder_path, scaling_factor):
    """
    Process all images in a folder to extract radii data.
    """
    angles_and_radii = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".JPG")])
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        angle = idx * 360 / len(image_files)
        angle, radii, center_x = process_image(image_path, angle, scaling_factor)
        angles_and_radii.append((angle, radii, center_x))
    return angles_and_radii

def save_to_csv(angles_and_radii, csv_filename="radii_data.csv"):
    """
    Saves the angle, radius, and height values to a CSV file.

    Parameters:
    angles_and_radii (list): A list of tuples where each tuple contains an angle,
                             a list of tuples with height and radius values, and a third value.
    csv_filename (str): The name of the CSV file to save the data. Default is 'radii_data.csv'.

    The CSV file will have columns: angle, radius, and z.
    """
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["angle", "radius", "z"])
        for angle, radii, _ in angles_and_radii:
            for height, radius in radii:
                writer.writerow([angle, radius, height])
                
def plot_3d_radii(angles_and_radii):
    """
    Visualize the measured radii in a 3D space and compare them to an ideal cylinder.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_height = max(height for angle, radii, _ in angles_and_radii for height, _ in radii)

    for angle_deg, radii, center_x in angles_and_radii:
        angle_rad = np.radians(angle_deg)
        x_points = []
        y_points = []
        z_points = []

        for height, radius in radii:
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            z = height
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
        
        # Plot points without connecting lines
        ax.scatter(x_points, y_points, z_points, color='r')

    # Plot the ideal cylinder
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, max_height, 100)  # Height dynamically set
    theta, z = np.meshgrid(theta, z)
    x = 4 * np.cos(theta)  # Radius = 4 mm
    y = 4 * np.sin(theta)  # Radius = 4 mm
    ax.plot_surface(x, y, z, alpha=0.3, color='b')

    # Create custom legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=5, label='Measured Points')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=5, label='Ideal Cylinder')
    ax.legend(handles=[red_patch, blue_patch])

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Height (mm)')
    plt.title('3D Radii Plot')
    plt.show()

def plot_radii_vs_height(angles_and_radii):
    """
    Show the average radius at different heights along the cylinder and compare it to the ideal radius (4mm).
    """
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

    avg_radii_array = np.array(avg_radii)
    heights_array = np.array(heights)

    ideal_radii = np.full_like(avg_radii_array, 4)  # Radius = 4 mm

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(heights_array, avg_radii_array, color='r', label='Measured Radius')
    ax.plot(heights_array, ideal_radii, color='b', linestyle='--', label='Ideal Radius')

    ax.set_xlabel('Height (mm)')
    ax.set_ylabel('Radius (mm)')
    ax.set_title('Average Radius vs Height')
    ax.legend()

    plt.show()

    errors = ((avg_radii_array - ideal_radii) / 4) * 100  # Calculate relative error in percentage
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"Mean error: {mean_error:.2f}%")
    print(f"Max error: {max_error:.2f}%")

    return errors

def validate_with_ideal_cylinder(angles_and_radii):
    """
    Calculate and visualize the error between measured radii and the ideal cylinder.
    """
    ideal_radius = 4  # mm

    errors = []
    heights = []
    for angle, radii, _ in angles_and_radii:
        for height, radius in radii:
            ideal_radius_at_height = ideal_radius
            error = ((euclidean([ideal_radius_at_height], [radius])) / ideal_radius_at_height) * 100
            errors.append(error)
            heights.append(height)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    # Calculate RMSE
    squared_errors = np.square(errors)
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)

    # Calculate standard deviation
    std_dev = np.std(errors)

    print(f"Mean error compared to ideal cylinder: {mean_error:.2f}%")
    print(f"Max error compared to ideal cylinder: {max_error:.2f}%")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}%")
    print(f"Standard Deviation of Errors: {std_dev:.2f}%")

    # Plot errors as scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(heights, errors, color='r', label='Error')
    ax.set_xlabel('Height (mm)')
    ax.set_ylabel('Error (%)')
    ax.set_title('Error Distribution')
    ax.legend()
    plt.show()

    # Plot histogram of errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=30, color='blue', edgecolor='black')
    ax.set_xlabel('Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Errors')
    plt.show()

if __name__ == "__main__":
    folder_path = "/NewTestImages/"
    angles_and_radii = process_folder(folder_path, scaling_factor)
    
    # Generate and display intermediate plots
    sample_image_path = os.path.join(folder_path, "image_3.JPG")
    try:
        edges, mask_blue_tape, sample_image = segment_specimen(sample_image_path)

        # Display the masked image
        plt.imshow(mask_blue_tape, cmap='gray')
        plt.title("Masked Image")
        plt.axis('off')
        plt.show()

        # Display the edge-detected image
        edges_thick = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        plt.imshow(edges_thick, cmap='gray')
        plt.title("Edge-detected Image")
        plt.axis('off')
        plt.show()

        # Calculate and display the centerline
        _, center_x = calculate_centerline_and_radii(edges, mask_blue_tape, scaling_factor)
        centerline_image = sample_image.copy()
        center_x = int(center_x)  # Ensure center_x is an integer
        cv2.line(centerline_image, (center_x, 0), (center_x, centerline_image.shape[0]), (0, 255, 0), 5)
        
        # Draw blue tape contours
        contours, _ = cv2.findContours(mask_blue_tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(centerline_image, contours, -1, (255, 0, 0), 5)
        
        plt.imshow(cv2.cvtColor(centerline_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Centerline and Blue Tape Contour")
        plt.axis('off')
        plt.show()

        # Calculate radii and plot them
        _, radii, _ = process_image(sample_image_path, 0, scaling_factor)
        heights, radii_values = zip(*radii)
        plt.scatter(heights, radii_values, label="Measured Radius", color='r')
        plt.xlabel("Height (mm)")
        plt.ylabel("Radius (mm)")
        plt.title("Radius Measurement")
        plt.legend()
        plt.show()
    except FileNotFoundError as e:
        print(e)

    plot_3d_radii(angles_and_radii)
    errors = plot_radii_vs_height(angles_and_radii)
    validate_with_ideal_cylinder(angles_and_radii)
    save_to_csv(angles_and_radii)
