import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_silhouette(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 168]), np.array([172, 111, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def calculate_centerline_radii(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest_contour)
    center_x = w // 2
    radii = []
    for i in range(h):
        row = mask[i]
        if np.sum(row) > 0:
            left = np.min(np.where(row > 0))
            right = np.max(np.where(row > 0))
            radius = (right - left) / 2
            radii.append((i, center_x - left, radius))
    return radii

def radii_to_3d(radii, angle, angle_increment):
    points_3d = []
    for height, _, radius in radii:  # Adjusted to ignore the center in this context
        angle_rad = np.radians(angle)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        points_3d.append([x, y, height])
    return points_3d

def main():
    image_dir = "/Users/vamsikrishna/Desktop/TestImages"
    image_paths = [os.path.join(image_dir, f"image_{i}.jpg") for i in range(24)]
    angle_increment = 360 / len(image_paths)
    all_points_3d = []

    for idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        mask = extract_silhouette(image_path)
        radii = calculate_centerline_radii(mask)
        points_3d = radii_to_3d(radii, idx * angle_increment, angle_increment)
        all_points_3d.extend(points_3d)

    # Visualization
    points_3d_np = np.array(all_points_3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d_np[:,0], points_3d_np[:,1], points_3d_np[:,2])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

if __name__ == "__main__":
    main()
