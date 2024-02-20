import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_specimen(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 205])
    upper_white = np.array([172, 171, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return image, mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def calculate_centerline_radii(contour, image):
    # Calculate the moments of the contour to find the center
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    
    # Draw the contour and the centerline on the image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.line(image, (cx, 0), (cx, image.shape[0]), (255, 0, 0), 2)
    
    # Calculate the radii from the centerline to the contour
    radii = []
    for point in contour:
        x, y = point[0]
        if x < cx:  # Left side of the centerline
            radius = cx - x
        else:  # Right side of the centerline
            radius = x - cx
        radii.append((y, radius))
    
    # Return the radii and the image with contour and centerline drawn
    return radii, image


def plot_radii(radii):
    # Plot the radii
    ys, radii_values = zip(*radii)
    plt.plot(ys, radii_values)
    plt.xlabel('Height')
    plt.ylabel('Radius')
    plt.title('Radius at Each Height')
    plt.show()

def main():
    image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'
    image, mask = segment_specimen(image_path)
    
    largest_contour = find_largest_contour(mask)
    radii, image_with_contour_centerline = calculate_centerline_radii(largest_contour, image)
    
    # Show the image with contour and centerline
    cv2.imshow('Contour with Centerline', image_with_contour_centerline)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Plot the radii
    plot_radii(radii)

if __name__ == '__main__':
    main()
