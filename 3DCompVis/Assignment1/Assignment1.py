import cv2
import numpy as np
import random
from itertools import combinations

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1.flatten()
    x3, y3, x4, y4 = line2.flatten()
    
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1
    
    if determinant == 0:
        return None
    else:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return (int(x), int(y))

def distance_point_to_line(point, line):
    x0, y0 = point
    x1, y1, x2, y2 = line.flatten()
    distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distance

def reestimate_vanishing_point(inlier_lines):
    intersection_points = []
    for line1, line2 in combinations(inlier_lines, 2):
        intersection = calculate_intersection(line1, line2)
        if intersection:
            intersection_points.append(intersection)
    
    if intersection_points:
        x_coords, y_coords = zip(*intersection_points)
        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        return (avg_x, avg_y)
    else:
        return None

def extend_line(line, vanishing_point, length=1000):
    x1, y1, x2, y2 = line.flatten()
    
    dx, dy = x2 - x1, y2 - y1
    norm = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / norm, dy / norm
    
    pt1 = (int(x1 - dx * length), int(y1 - dy * length))
    pt2 = (int(x2 + dx * length), int(y2 + dy * length))
    
    return pt1, pt2

def main():
    image = cv2.imread("ELTECar3.png", 0)
    if image is None:
        print("Image not found.")
        return

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(image)[0]

    if lines is None or len(lines) < 2:
        print("Not enough lines detected to find intersections.")
        return

    threshold_distance = 10
    num_iterations = 100

    best_inliers = []
    best_candidate_point = None

    for _ in range(num_iterations):
        line1, line2 = random.sample(list(lines), 2)
        candidate_point = calculate_intersection(line1, line2)
        if not candidate_point:
            continue

        inliers = []
        for line in lines:
            dist = distance_point_to_line(candidate_point, line)
            if dist < threshold_distance:
                inliers.append(line)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_candidate_point = candidate_point

    final_vanishing_point = reestimate_vanishing_point(best_inliers) if best_inliers else None

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line in best_inliers:
        pt1, pt2 = extend_line(line, final_vanishing_point)
        cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

    for line in lines:
        if not any(np.array_equal(line, inlier_line) for inlier_line in best_inliers):
            x1, y1, x2, y2 = line.flatten()
            cv2.line(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    if final_vanishing_point:
        cv2.circle(color_image, final_vanishing_point, 15, (0, 0, 255), -1)

    cv2.imshow("Vanishing Point Detection", color_image)
    cv2.imwrite("vanishing_point_extended_inlier_lines.png", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
