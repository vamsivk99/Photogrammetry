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

def calculate_centerline_radii(contour, image, mask):
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    
    # Draw the centerline on the mask
    cv2.line(mask, (cx, 0), (cx, image.shape[0]), (255, 0, 0), 2)
    cv2.imshow('Centerline', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Calculate radii
    radii = []
    for point in contour:
        x, y = point[0]
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radii.append((y, radius))
    
    return radii, mask

def main():
    image_dir = '/Users/vamsikrishna/Desktop/TestImages/'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPG')]
    print("Number of images:", len(image_paths))
    all_radii = []
    for image_path in image_paths:
        image, mask = segment_specimen(image_path)
        largest_contour = find_largest_contour(mask)
        radii, image_with_contour_centerline = calculate_centerline_radii(largest_contour, image, mask)
        all_radii.append(radii)
    
    # Now plot all radii together
    for radii in all_radii:
        ys, radii_values = zip(*radii)
        plt.scatter(ys, radii_values, alpha=0.5)  # Use scatter plot for clarity
    
    plt.xlabel('Height')
    plt.ylabel('Radius')
    plt.title('Radius at Each Height for All Images')
    plt.show()

if __name__ == '__main__':
    main()
