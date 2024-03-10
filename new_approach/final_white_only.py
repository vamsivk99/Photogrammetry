import numpy as np
import cv2

# Load the image from file
image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'  # Replace with your image file path
image_cv2 = cv2.imread(image_path)

# Convert the image to HSV
image_hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)

# Define HSV ranges for white, orange, and blue colors
white_lower = np.array([0, 0, 0], dtype=np.uint8)
white_upper = np.array([0,0,255], dtype=np.uint8)
orange_lower = np.array([5, 50, 50], dtype=np.uint8)
orange_upper = np.array([20, 255, 255], dtype=np.uint8)
blue_lower = np.array([10, 50, 50], dtype=np.uint8)
blue_upper = np.array([130, 255, 255], dtype=np.uint8)

# Create masks for the colors
mask_white = cv2.inRange(image_hsv, white_lower, white_upper)
mask_orange = cv2.inRange(image_hsv, orange_lower, orange_upper)
mask_blue = cv2.inRange(image_hsv, blue_lower, blue_upper)

# Combine masks to isolate colors to remove (orange and blue)
mask_remove = cv2.bitwise_or(mask_orange, mask_blue)

# Invert the remove mask to isolate colors to keep, then combine with white mask
mask_keep = cv2.bitwise_not(mask_remove)
mask_final = cv2.bitwise_and(mask_white, mask_keep)

# Apply the final mask to the image
result_white_only = cv2.bitwise_and(image_cv2, image_cv2, mask=mask_final)

# Save the final image
output_filename = 'final_white_only_image.png'  # Set your output file name
cv2.imwrite(output_filename, result_white_only)

# Display the final image
cv2.imshow('Final Image', result_white_only)
cv2.waitKey(0)
cv2.destroyAllWindows()