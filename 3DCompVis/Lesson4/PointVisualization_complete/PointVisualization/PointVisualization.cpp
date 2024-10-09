#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#include "Camera.h"

// std::vector<cv::Point3f> readPLY(const std::string& fileName)

std::vector<cv::Mat> readPointCloud(const std::string& fileName) {
	// .xyz files only

	std::vector<cv::Mat> pc;

	std::ifstream f(fileName);
	if (!f.is_open()) {
		std::cout << "Couldn't open file!" << std::endl;
		return pc;
	}

	double x, y, z;
	while (!f.eof()) {
		f >> x >> y >> z;
		cv::Mat newPoint(3, 1, CV_64F);
		newPoint.at<double>(0, 0) = x;
		newPoint.at<double>(1, 0) = y;
		newPoint.at<double>(2, 0) = z;
		pc.push_back(newPoint);
	}

	return pc;
}

void drawPoints(cv::Mat& img, const std::vector<cv::Mat>& pc, const cv::Mat& K, const cv::Mat& R, const cv::Mat& T) {
	for (cv::Mat p : pc) {
		cv::Mat p_C = R * (p - T); // World -> Camera

		cv::Scalar color(255, 255, 255);
		if (p_C.at<double>(2, 0) < 0.0) color = cv::Scalar(255, 0, 0); // Behind camera

		p_C = K * p_C; // Projection
		p_C /= p_C.at<double>(2, 0); // Homogeneous divison

		cv::circle(img, cv::Point((int)p_C.at<double>(0, 0), (int)p_C.at<double>(1, 0)), 2, color, cv::FILLED);


		// Projection with OpenCV function:
		// https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
		// cv::projectPoints(objectPoints, rvec, tvec, K, distCoeffs, imagePoints)
		//   -  objectPoints - std::vector<cv::Point3f> (input)
		//   -  imagePoints - std::vector<cv::Point2f> (output)
		//   -  rvec - Rotation vector (see: cv::Rodrigues https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac)
		//   -  distCoeffs - can be empty Mat
	}
}

int main() {

	std::string pc_file = "PointClouds/Structure.xyz";
	std::vector<cv::Mat> pc = readPointCloud(pc_file);
	cv::Mat img = cv::Mat::zeros(600, 800, CV_8UC3);

	Camera cam(1000, 1000, 400, 300);

	double phi = 0.5;
	double theta = 1.0;
	double r = 100.0;
	cam.setExtrinsics(phi, theta, r);

	char key;
	while (true) {
		key = cv::waitKey(0);
		if (key == 27) break;

		switch (key) {
		case 'd':
			phi += 0.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		case 'a':
			phi -= 0.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		case 'w':
			theta += 0.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		case 's':
			theta -= 0.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		case 'q':
			r *= 1.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		case 'e':
			r /= 1.1;
			cam.setExtrinsics(phi, theta, r);
			break;
		}

		// Display updated image
		img = cv::Mat::zeros(600, 800, CV_8UC3);
		drawPoints(img, pc, cam.getIntrinsics(), cam.getRotation(), cam.getTranslation());
		cv::imshow("Display window", img);
	}

}
