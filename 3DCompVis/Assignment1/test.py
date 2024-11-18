import cv2
import numpy as np

# Load or create an image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Define two points
point1 = (100, 200)
point2 = (400, 300)

# Calculate the slope
if point2[0] - point1[0] != 0:  # Avoid division by zero
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = point1[1] - slope * point1[0]

    # Extend line to the image borders
    y1 = int(slope * 0 + intercept)  # y at x = 0 (left border)
    y2 = int(slope * image.shape[1] + intercept)  # y at x = width (right border)

    # Draw the extended line
    cv2.line(image, (0, y1), (image.shape[1], y2), (0, 255, 0), 2)
else:
    # If the line is vertical, draw from top to bottom border
    cv2.line(image, (point1[0], 0), (point1[0], image.shape[0]), (0, 255, 0), 2)

# Display the image
cv2.imshow("Extended Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
