import cv2
import numpy as np
import os

def create_mask_for_color_range(image, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def subtract_masks_from_image(image, masks):
    combined_mask = np.zeros(image.shape[:2], dtype="uint8")
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    inverted_mask = cv2.bitwise_not(combined_mask)
    result = cv2.bitwise_and(image, image, mask=inverted_mask)
    return result

def find_and_draw_centerline(image, mask_blue):
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate the center of the rectangle (blue tape's assumed center)
        center_x = int((box[0][0] + box[2][0]) / 2)
        cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (0, 255, 0), 2)
    return image

def process_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".JPG"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Define color ranges for the orangish background and blue tape
            lower_orange = np.array([5, 50, 50])
            upper_orange = np.array([15, 255, 255])
            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])

            # Create masks
            mask_orange = create_mask_for_color_range(image, lower_orange, upper_orange)
            mask_blue = create_mask_for_color_range(image, lower_blue, upper_blue)

            # Subtract masks from image to isolate the specimen
            result_image = subtract_masks_from_image(image, [mask_orange, mask_blue])

            # Find and draw centerline
            result_with_centerline = find_and_draw_centerline(result_image, mask_blue)

            cv2.imshow("Result", result_with_centerline)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    images_directory = '/Users/vamsikrishna/Desktop/TestImages/'
    process_images_in_directory(images_directory)
