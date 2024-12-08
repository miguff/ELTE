import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import sys




def main():
    # Load images
    folder = 1 #folder selector
    lst = os.listdir(f"{folder}")
    number_files = len(lst)
    print(number_files) #check the number of files

    newimage = None #This will check whether we have had our firt iteration or not
    for i in range(min(3, number_files)-1):

        #choose image
        if newimage == None:
            image1_path = lst[i]
            image1 = cv2.imread(f"{folder}/{image1_path}")
        else:
            image1_path = newimage
            image1 = cv2.imread(f"{image1_path}")
        image2_path = lst[i+1]
        image2 = cv2.imread(f"{folder}/{image2_path}")

        #Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        #SIFT initialization
        sift = cv2.SIFT_create()

        # We them need to create the keypoints and the descriptors for the feature matching algorithm
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Use the FlannBasedMatcher
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        #Find the good matcher
        good_matches = [m for m, n in matches if m.distance < 0.5* n.distance]

        #then Extract the points
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        #Choose whether we want to implement the cv2 built in method or the one that we did on the class
        lessontrigger = True
        if lessontrigger:
            H = UseLessonHomography(points1, points2)
            name = "lesson"
        else:
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            name = "cv2"

        #create the canvas for both the pictures
        canvas_width = image1.shape[1] + image2.shape[1]
        canvas_height = max(image1.shape[0], image2.shape[0])
        result = cv2.warpPerspective(image2, H, (canvas_width*1, canvas_height*1))
        
        #Insert the first image to the canvas
        result[0:image1.shape[0], 0:image1.shape[1]] = image1

        #Create the save name
        stitched_image_path = f"{folder}_{name}_homography{i}{i+1}.png"
        newimage = stitched_image_path


        # Delete the black parts, that stops us from making the good third picture
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        #Find the threshold and create a binary mask for them
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        #Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Get the larges contours box coordinates
        x, y, w, h = cv2.boundingRect(contours[0])

        #Crop the image
        cropped_image = result[y:y+h, x:x+w]

        #Save the image, for the next iter of for final
        cv2.imwrite(stitched_image_path, cropped_image)

        # Display the stitched image
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
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


def UseLessonHomography(points1, points2):
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    A = build_A(points1_norm, points2_norm)

    # Solve for homography using SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    # Denormalize homography
    H = np.linalg.inv(T1) @ H_norm @ T2


    return H 

# Build matrix A for solving homography
def build_A(points1, points2):
    A = []
    for (x1, y1), (x2, y2) in zip(points1, points2):
        A.append([-x2, -y2, -1, 0, 0, 0, x2 * x1, y2 * x1, x1])
        A.append([0, 0, 0, -x2, -y2, -1, x2 * y1, y2 * y1, y1])
    return np.array(A)


if __name__ == "__main__":
    main()
