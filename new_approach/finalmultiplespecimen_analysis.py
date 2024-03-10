import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def process_image(image_path):
    segmented_image, mask_blue, original_image = segment_specimen(image_path)
    
    # Display the segmented image
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Specimen")
    plt.show()
    
    # Calculate and display the centerline
    image_with_centerline = calculate_centerline(original_image.copy(), mask_blue)
    plt.imshow(cv2.cvtColor(image_with_centerline, cv2.COLOR_BGR2RGB))
    plt.title("Image with Centerline")
    plt.show()

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG"):  # change it based on your file extensions
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            process_image(image_path)

if __name__ == "__main__":
    folder_path = "/Users/vamsikrishna/Desktop/TestImages/"
    process_folder(folder_path)
