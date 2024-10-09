#pragma once

#include <opencv2/core.hpp>

class Camera
{
public:
	Camera(double fku, double fkv, double u0, double v0);

	void setExtrinsics(double phi, double theta, double r);

	cv::Mat getIntrinsics() const { return K; }
	cv::Mat getRotation() const { return R; }
	cv::Mat getTranslation() const { return T; }

private:
	cv::Mat K;
	cv::Mat R;
	cv::Mat T;
};

