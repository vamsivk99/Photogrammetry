import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image from the provided file path
image_path = '/Users/vamsikrishna/Desktop/DSC00250.JPG'
image_cv2 = cv2.imread(image_path)

# Convert the image to HSV
image_hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)

# Define HSV ranges for orange and blue colors to remove
orange_lower = np.array([5, 20, 50], dtype=np.uint8)
orange_upper = np.array([15, 255, 240], dtype=np.uint8)
blue_lower = np.array([35, 0, 0], dtype=np.uint8)
blue_upper = np.array([120, 255, 255], dtype=np.uint8)

# Create masks for orange and blue colors
mask_orange = cv2.inRange(image_hsv, orange_lower, orange_upper)
mask_blue = cv2.inRange(image_hsv, blue_lower, blue_upper)

# Combine masks for the colors to remove
mask_remove = cv2.bitwise_or(mask_orange, mask_blue)

# Invert mask to keep the rest of the image
mask_keep = cv2.bitwise_not(mask_remove)

# Apply the mask to highlight the specimen
result_specimen_highlighted = cv2.bitwise_and(image_cv2, image_cv2, mask=mask_keep)

# Convert to RGB for displaying with matplotlib
result_specimen_highlighted_rgb = cv2.cvtColor(result_specimen_highlighted, cv2.COLOR_BGR2RGB)
image_hsv_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
mask_orange_rgb = cv2.cvtColor(mask_orange, cv2.COLOR_GRAY2RGB)
mask_blue_rgb = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB)
mask_remove_rgb = cv2.cvtColor(mask_remove, cv2.COLOR_GRAY2RGB)
mask_keep_rgb = cv2.cvtColor(mask_keep, cv2.COLOR_GRAY2RGB)

# Plot the figures
# fig, ax = plt.subplots(3, 2, figsize=(10, 15))

# ax[0, 0].imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
# ax[0, 0].set_title('Original Image')
# ax[0, 0].axis('off')

# ax[0, 1].imshow(image_hsv_rgb)
# ax[0, 1].set_title('HSV Image')
# ax[0, 1].axis('off')

# ax[1, 0].imshow(mask_orange_rgb)
# ax[1, 0].set_title('Mask for Orange')
# ax[1, 0].axis('off')

# ax[1, 1].imshow(mask_blue_rgb)
# ax[1, 1].set_title('Mask for Blue')
# ax[1, 1].axis('off')

# ax[2, 0].imshow(mask_remove_rgb)
# ax[2, 0].set_title('Combined Mask to Remove Colors')
# ax[2, 0].axis('off')

# ax[2, 1].imshow(result_specimen_highlighted_rgb)
# ax[2, 1].set_title('Specimen Highlighted (Final Image)')
# ax[2, 1].axis('off')

# plt.tight_layout()
# plt.show()

# Plot the figures
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# plt.figure(figsize=(6, 6))
# plt.imshow(image_hsv_rgb)
# plt.title('HSV Image')
# plt.axis('off')
# plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(mask_orange_rgb)
plt.title('Mask for Orange')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(mask_blue_rgb)
plt.title('Mask for Blue')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(mask_remove_rgb)
plt.title('Combined Mask to Remove Colors')
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(result_specimen_highlighted_rgb)
plt.title('Specimen Highlighted (Final Image)')
plt.axis('off')
plt.show()