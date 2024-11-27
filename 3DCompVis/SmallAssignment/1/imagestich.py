import cv2
import numpy as np
import matplotlib.pyplot as plt





def main():
    # Load images
    image1_path = "DSCF8660.JPG"
    image2_path = "improved_manual_homography616263.png"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match descriptors using BFMatcher with Lowe's ratio test
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.5* n.distance]

    # Extract matched points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    A = build_A(points1_norm, points2_norm)

    # Solve for homography using SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Denormalize homography
    H = np.linalg.inv(T1) @ H_norm @ T2

    # Create a canvas large enough for both images
    canvas_width = image1.shape[1] + image2.shape[1]
    canvas_height = max(image1.shape[0], image2.shape[0])
    result = cv2.warpPerspective(image1, H, (canvas_width*1, canvas_height*1))
    
    # Overlay the first image onto the canvas
    result[0:image2.shape[0], 0:image2.shape[1]] = image2

    # Save and display the stitched result
    stitched_image_path = "improved_manual_homography60616263.png"
    cv2.imwrite(stitched_image_path, result)

    # Display the stitched image
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Image with Manual Homography")
    plt.axis('off')
    plt.show()

# Normalize points
def normalize_points(points):
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    T = np.array([
        [1 / std_dev[0], 0, -mean[0] / std_dev[0]],
        [0, 1 / std_dev[1], -mean[1] / std_dev[1]],
        [0, 0, 1]
    ])
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (T @ points_h.T).T
    return normalized_points[:, :2], T



# Build matrix A for solving homography
def build_A(points1, points2):
    A = []
    for (x1, y1), (x2, y2) in zip(points1, points2):
        A.append([-x2, -y2, -1, 0, 0, 0, x2 * x1, y2 * x1, x1])
        A.append([0, 0, 0, -x2, -y2, -1, x2 * y1, y2 * y1, y1])
    return np.array(A)


if __name__ == "__main__":
    main()
