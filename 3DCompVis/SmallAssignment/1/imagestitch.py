import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the two images
image1_path = r"C:\Users\msi\Documents\ELTE\3DCompVis\SmallAssignment\1\DSCF8661.JPG"  # Replace with your image paths
image2_path = r"C:\Users\msi\Documents\ELTE\3DCompVis\SmallAssignment\1\DSCF8662.JPG"
image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

# Convert images to grayscale for SIFT
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)


# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance

print(matches)

# Extract point pairs
points1 = [keypoints1[m.queryIdx].pt for m in matches]
points2 = [keypoints2[m.trainIdx].pt for m in matches]

# Create a pandas DataFrame for the matched points
df_points = pd.DataFrame({
    "Image1_X": [pt[0] for pt in points1],
    "Image1_Y": [pt[1] for pt in points1],
    "Image2_X": [pt[0] for pt in points2],
    "Image2_Y": [pt[1] for pt in points2]
})

print(df_points)


# Function to calculate the normalization matrix
def calculate_normalization_matrix(points):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std_avg = np.mean(std)
    scale = np.sqrt(2) / std_avg
    offset = -scale * mean
    T = np.array([[scale, 0, offset[0]],
                  [0, scale, offset[1]],
                  [0, 0, 1]])
    return T

# Function to apply the normalization matrix to the points
def normalize_points(points, T):
    normalized_points = []
    for point in points:
        point_homogeneous = np.append(point, 1)
        normalized_point = T @ point_homogeneous
        normalized_points.append(normalized_point[:2] / normalized_point[2])
    return np.array(normalized_points)

# Function to calculate the homography matrix
def calculate_homography(point_pairs):
    A = []
    for point1, point2 in point_pairs:
        x, y = point1[0], point1[1]
        xp, yp = point2[0], point2[1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    
    #U, S, Vt = np.linalg.svd(A)
    
    eig_val, eig_vec = np.linalg.eig(A.T@A)
    
    
    Vt = eig_vec[:,np.argmin(eig_val)]
    
    H = Vt.reshape(3, 3)
    
    return H / H[2, 2]


# points1 = df_points[:,:2]
# points2 = df_points[:,-2:]

# Normalize the points and calculate the homography matrix
T1 = calculate_normalization_matrix(points1)
T2 = calculate_normalization_matrix(points2)

normalized_points1 = normalize_points(points1, T1)
normalized_points2 = normalize_points(points2, T2)

normalized_point_pairs = list(zip(points1, points2))
H = calculate_homography(normalized_point_pairs)

# Denormalize the homography matrix
#H = np.linalg.inv(T2) @ H_normalized @ T1

# Read the images
# img1 = cv2.imread('b.png')
# img2 = cv2.imread('a.png')


points1 = np.array(points1, dtype=np.int32)
points2 = np.array(points2, dtype=np.int32)


img1 = image1
img2 = image2

# Warp the second image to the first using the homography matrix
img2_transformed = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# Stitch the images together
stitched_img = img2_transformed
stitched_img[0:img1.shape[0], 0:img1.shape[1]] = img1

# Save the stitched image
cv2.imwrite('stitched_image.png', stitched_img)

# Display the stitched image
cv2.imshow('Stitched Image', stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()