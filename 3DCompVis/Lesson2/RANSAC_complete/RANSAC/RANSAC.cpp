// RANSAC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


//Customary OpenCV import
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <random>
#include <ctime>
#include <vector>

int WIDTH = 640;
int HEIGHT = 480;

std::mt19937 rng;
std::uniform_real_distribution<float> udist(0.f, 1.f);
std::normal_distribution<float> ndist(0.f, 10.f);

void init_random()
{
    rng.seed(static_cast<long unsigned int>(time(0)));
}

float random_number()
{
    return udist(rng);
}

float random_noise()
{
    return ndist(rng);
}

std::vector<cv::Point2f> generateData(int N, cv::Point2f p, cv::Point2f dir, float inlier_ratio)
{
    std::vector<cv::Point2f> ret;

    int N_inlier = N * inlier_ratio;
    int N_outlier = N - N_inlier;

    for (int i = 0; i < N_inlier; ++i)
    {
        float diag = sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT);

        float t = random_number() * diag - diag / 2.f;
        cv::Point2f q = p + t * dir;
        q.x += random_noise();
        q.y += random_noise();

        ret.push_back(q);
    }

    for (int i = 0; i < N_outlier; ++i)
    {
        cv::Point2f q;
        q.x = random_number() * WIDTH;
        q.y = random_number() * HEIGHT;
        ret.push_back(q);
    }

    return ret;
}

cv::Point3f fitLine(std::vector<cv::Point2f>& points)
{
    cv::Mat X(points.size(), 3, CV_32F);
    for (int i = 0; i < points.size(); ++i)
    {
        X.at<float>(i, 0) = points[i].x;
        X.at<float>(i, 1) = points[i].y;
        X.at<float>(i, 2) = 1.f;
    }

    cv::Mat XtX = X.t() * X;

    cv::Mat evals, evecs;
    cv::eigen(XtX, evals, evecs);

    cv::Point3f l;
    l.x = evecs.at<float>(2, 0);
    l.y = evecs.at<float>(2, 1);
    l.z = evecs.at<float>(2, 2);

    // Normalization
    float n = sqrt(l.x * l.x + l.y * l.y);
    l /= n;

    return l;
}

cv::Point3f fitLine(cv::Point2f p1, cv::Point2f p2)
{
    cv::Point2f dir = p2 - p1;
    dir /= sqrt(dir.x * dir.x + dir.y * dir.y); // normalization

    cv::Point3f l;
    l.x = -dir.y;
    l.y = dir.x;
    l.z = -l.x * p1.x - l.y * p1.y;

    return l;
}

float distance(cv::Point3f& l, cv::Point2f p)
{
    return abs(l.x * p.x + l.y * p.y + l.z);
}

int iterationNumber(float confidence, int sampleSize, int inliers, int N)
{
    float a = std::log(1.0f - confidence);
    float ratio = (float)inliers / (float)N;
    float b = std::log(1.0f - std::pow(ratio, sampleSize));

    //if (b < std::numeric_limits<float>::epsilon()) return std::numeric_limits<int>::max();

    return std::ceil(a / b);
}

cv::Point3f RANSAC(std::vector<cv::Point2f>& points, int max_iterations, float sigma)
{
    int best_inliers_nr = 0;
    cv::Point3f best_model;
    std::vector<cv::Point2f> best_inliers;
    std::cout << "New max iters: " << max_iterations << std::endl;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // 1. Select m points randomly
        int idx1 = random_number() * (points.size() - 1);
        int idx2 = random_number() * (points.size() - 1);
        while (idx1 == idx2)
        {
            idx2 = random_number() * (points.size() - 1);
        }

        // 2. Fit a model on the sample
        cv::Point3f current_model = fitLine(points[idx1], points[idx2]);

        // 3. Calculating the distance of the points
        int inliers_nr = 0;
        std::vector<cv::Point2f> current_inliers;
        for (int i = 0; i < points.size(); ++i)
        {
            if (distance(current_model, points[i]) < sigma)
            {
                current_inliers.push_back(points[i]);
                ++inliers_nr;
            }
        }

        // 4. Save the best model so far
        if (inliers_nr > best_inliers_nr)
        {
            best_inliers_nr = inliers_nr;
            best_model = current_model;
            best_inliers = current_inliers;

            max_iterations = std::min(max_iterations, iterationNumber(0.95, 2, best_inliers_nr, points.size()));
            //std::cout << "New max iters: " << max_iterations << std::endl;
        }
    }

    best_model = fitLine(best_inliers);
    return best_model;
}

int main()
{
    init_random(); //We create random point, or just the randomness
    cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3); //Create an image


    //################# Generate a random line ########################################
    cv::Point2f p = { HEIGHT / 2.f, WIDTH / 2.f };
    float alpha = random_number() * CV_2PI;
    cv::Point2f dir;
    dir.x = cos(alpha);
    dir.y = sin(alpha); //We get the direction vector with the sin and cos alpha
    std::vector<cv::Point2f> points = generateData(1000, p, dir, 0.5f);

    for (int i = 0; i < points.size(); ++i)
    {
        cv::circle(img, points[i], 2, { 255, 255, 255 }, cv::FILLED);
    }

    cv::Point2f p1, p2;
    p1 = p - 1000.f * dir;
    p2 = p + 1000.f * dir;
    cv::line(img, p1, p2, { 255, 0, 0 }, 2);

    cv::Point3f l = RANSAC(points, 2000, 10.f);
    //cv::Point3f l = fitLine(points);
    p1.x = 0;
    p1.y = -l.z / l.y;

    p2.x = WIDTH;
    p2.y = (-l.z - l.x * p2.x) / l.y;

    cv::line(img, p1, p2, { 0, 255, 0 }, 2);

    cv::imshow("RANSAC", img);
    int key = cv::waitKey(0);

    return 0;
}

