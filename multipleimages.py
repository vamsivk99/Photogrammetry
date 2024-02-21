import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_specimen(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 205])
    upper_white = np.array([175, 181, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return image, mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def draw_contour(image, contour):
    # Draw the largest contour on the image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.imshow('Largest Contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

SCALE = 0.1  # assuming each pixel represents 0.1 cm

# def calculate_centerline_radii(contour, image, mask, scale=SCALE):
#     moments = cv2.moments(contour)
#     cx = int(moments['m10']/moments['m00'])
#     cy = int(moments['m01']/moments['m00'])
    
#     # Draw the centerline on the mask
#     cv2.line(mask, (cx, 0), (cx, image.shape[0]), (255, 0, 0), 2)
#     cv2.imshow('Centerline', mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     # Calculate radii
#     radii = []
#     for point in contour:
#         x, y = point[0]
#         radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
#         radii.append((y, radius))
#         # Convert pixel radii to centimeters
#     radii_cm = [(y, radius * scale) for y, radius in radii]
    
#     return radii_cm, mask
def calculate_centerline_radii(contour, image, draw_centerline=True):
    # Calculate the moments of the largest contour
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    
    # Create an empty image to draw the contour filled (for visualization)
    contour_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_image, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Calculate the radii from the centerline to the contour
    radii = []
    for y in range(image.shape[0]):
        # Get the slice of the contour image at height y
        row = contour_image[y, :]
        if np.any(row):
            left_most = np.where(row > 0)[0][0]
            right_most = np.where(row > 0)[0][-1]
            radius_left = cx - left_most
            radius_right = right_most - cx
            # Use the average of the left and right radius for symmetry
            average_radius = (radius_left + radius_right) / 2
            radii.append((y, average_radius))
    
    # Draw the centerline for visualization
    if draw_centerline:
        for y, radius in radii:
            cv2.circle(image, (cx, y), 1, (0, 0, 255), -1)  # Draw small circles to represent the centerline
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw the largest contour
    
    # Show the image with the centerline and contour
    cv2.imshow('Image with centerline and contour', image)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    
    return radii

def main():
    image_dir = '/Users/vamsikrishna/Desktop/TestImages/'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPG')]
    print("Number of images:", len(image_paths))
    all_radii = []
    for image_path in image_paths:
        image, mask = segment_specimen(image_path)
        largest_contour = find_largest_contour(mask)
        radii = calculate_centerline_radii(largest_contour, image, True)
        all_radii.append(radii)
    
    # Now plot all radii together
    for radii in all_radii:
        ys, radii_values = zip(*radii)
        plt.scatter(ys, radii_values, alpha=0.5)  # Use scatter plot for clarity
    
    plt.xlim([0, 2100])  # Set x-axis range
    plt.ylim([0, 400])  # Set y-axis range
    plt.xlabel('Height')
    plt.ylabel('Radius')
    plt.title('Radius at Each Height for All Images')
    plt.show()

if __name__ == '__main__':
    main()
