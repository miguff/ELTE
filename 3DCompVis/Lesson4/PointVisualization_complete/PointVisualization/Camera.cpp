#include "Camera.h"

#include <cmath>

Camera::Camera(double fku, double fkv, double u0, double v0) {
	K = cv::Mat::eye(3, 3, CV_64F);
	K.at<double>(0, 0) = fku;
	K.at<double>(1, 1) = fkv;
	K.at<double>(0, 2) = u0;
	K.at<double>(1, 2) = v0;

	R = cv::Mat::eye(3, 3, CV_64F);
	T = cv::Mat::zeros(3, 1, CV_64F);
};

void Camera::setExtrinsics(double phi, double theta, double r) {
	double tx = cos(phi) * sin(theta);
	double ty = sin(phi) * sin(theta);
	double tz = cos(theta);

	T.at<double>(0, 0) = tx * r;
	T.at<double>(1, 0) = ty * r;
	T.at<double>(2, 0) = tz * r;

	cv::Point3d Z(-tx, -ty, -tz);
	Z /= cv::norm(Z);
	cv::Point3d Y(ty, -tx, 0.0f);
	Y /= cv::norm(Y);
	int HowManyPi = (int)floor(theta / 3.1415); // Theta between [0, pi)
	if (HowManyPi % 2 == 0) {
		Y *= -1;
	}
	cv::Point3d X = Y.cross(Z);

	R.at<double>(0, 0) = X.x;
	R.at<double>(0, 1) = X.y;
	R.at<double>(0, 2) = X.z;

	R.at<double>(1, 0) = Y.x;
	R.at<double>(1, 1) = Y.y;
	R.at<double>(1, 2) = Y.z;

	R.at<double>(2, 0) = Z.x;
	R.at<double>(2, 1) = Z.y;
	R.at<double>(2, 2) = Z.z;
}