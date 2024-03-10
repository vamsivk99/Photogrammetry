import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def exclude_colors(image, hsv_ranges_to_exclude):
    """Create a mask to exclude specific HSV color ranges."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    exclude_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create masks for unwanted colors and combine them
    for lower, upper in hsv_ranges_to_exclude:
        color_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        exclude_mask = cv2.bitwise_or(exclude_mask, color_mask)

    # Invert the mask to target the specimen
    specimen_mask = cv2.bitwise_not(exclude_mask)

    # Apply the specimen mask to get the isolated specimen
    isolated_specimen = cv2.bitwise_and(image, image, mask=specimen_mask)
    return isolated_specimen, specimen_mask

def main():
    image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'
    image = load_image(image_path)
    
    # Define HSV ranges to exclude (e.g., blue tape and orange background)
    hsv_ranges_to_exclude = [
        # Blue color range
        ([100, 150, 0], [140, 255, 255]),
        # Orange color range
        ([5, 50, 50], [15, 255, 255]),
        # Adjust these ranges based on your specific colors
    ]
    
    isolated_specimen, specimen_mask = exclude_colors(image, hsv_ranges_to_exclude)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(isolated_specimen, cv2.COLOR_BGR2RGB))
    plt.title("Isolated Specimen")
    
    plt.subplot(1, 2, 2)
    plt.imshow(specimen_mask, cmap='gray')
    plt.title("Specimen Mask")

    plt.show()

if __name__ == "__main__":
    main()
