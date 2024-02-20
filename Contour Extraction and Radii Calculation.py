import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_centerline_radii(contour, image_shape):
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    
    # Draw the centerline
    image_with_centerline = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.line(image_with_centerline, (cx, 0), (cx, image_shape[0]), (255, 0, 0), 2)
    cv2.imshow('Centerline', image_with_centerline)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Calculate radii
    radii = []
    for point in contour:
        x, y = point[0]
        radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radii.append((y, radius))
    
    return radii

def plot_radii(radii):
    # Plot radii
    ys, radii_values = zip(*radii)
    plt.plot(ys, radii_values)
    plt.xlabel('Height')
    plt.ylabel('Radius')
    plt.title('Radius at Each Height')
    plt.show()

def main():
    image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'
    image, mask = segment_specimen(image_path)
    
    # Show mask
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    largest_contour = find_largest_contour(mask)
    draw_contour(image, largest_contour)
    
    radii = calculate_centerline_radii(largest_contour, image.shape)
    plot_radii(radii)

if __name__ == '__main__':
    main()
