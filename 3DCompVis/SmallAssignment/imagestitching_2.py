import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1_path = "final_stitched_image_585960.png"
image2_path = "DSCF8661.JPG"
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match features using FLANN matcher
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Extract matched points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Find homography
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# Warp the second image using the homography matrix
height, width, _ = image1.shape
result = cv2.warpPerspective(image2, H, (width * 2, height*1))

stitched_image_path = "pre_final_stitched_image_58596061.png"
cv2.imwrite(stitched_image_path, result)
result[0:height, 0:width] = image1

# Display the stitched image
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.axis('off')
plt.show()

# Save the stitched image
stitched_image_path = "final_stitched_image_58596061.png"
cv2.imwrite(stitched_image_path, result)
