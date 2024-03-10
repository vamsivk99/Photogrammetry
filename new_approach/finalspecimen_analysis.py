import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_centerline(image, mask_blue):
    # Find contours in the blue mask
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the blue tape
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    
    # Draw the centerline on the image
    cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 2)
    return image

def calculate_radius_and_centerline(image, mask_blue):
    """Calculate the radius from the centerline to the edge of the blue tape."""
    # Find contours in the blue mask to identify the edges of the tape
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, image.shape[1] // 2  # Return None if no contour is found
    
    # Assuming the largest contour corresponds to the blue tape
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    
    # Assuming the tape's center is the rotational axis, calculate radius to the edge
    radius = w // 2
    return radius, center_x

def plot_radius(radius):
    """Plot the radius value."""
    # Simple plot with a single radius value
    plt.figure()
    plt.bar(['Radius'], [radius])
    plt.ylabel('Radius (in pixels)')
    plt.title('Radius from Centerline to Blue Tape Edge')
    plt.show()

def main(image_path):
    segmented_image, mask_blue, original_image = segment_specimen(image_path)
    
    # Calculate radius and centerline
    radius, center_x = calculate_radius_and_centerline(original_image, mask_blue)
    
    # Draw the centerline on the original image
    cv2.line(original_image, (center_x, 0), (center_x, original_image.shape[0]), (255, 0, 0), 2)
    
    # Show the original image with the centerline
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Centerline")
    plt.show()

    # Plot the calculated radius
    plot_radius(radius)

if __name__ == "__main__":
    image_path = "/Users/vamsikrishna/Desktop/DSC00250.JPG"
    main(image_path)