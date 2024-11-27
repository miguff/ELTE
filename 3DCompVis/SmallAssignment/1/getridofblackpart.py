import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image with black borders
image_with_borders_path = "final_stitched_image_5859.png"  # Replace with your file path
image_with_borders = cv2.imread(image_with_borders_path)

# Convert to grayscale for thresholding
gray = cv2.cvtColor(image_with_borders, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the image using the bounding box
cropped_image = image_with_borders[y:y+h, x:x+w]

# Save and display the cropped image
cropped_image_path = "cropped_image5859.png"  # Specify the output file path
cv2.imwrite(cropped_image_path, cropped_image)

# Display the cropped image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image Without Black Borders")
plt.axis('off')
plt.show()
