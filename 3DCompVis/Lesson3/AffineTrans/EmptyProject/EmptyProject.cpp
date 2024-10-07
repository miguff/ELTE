// EmptyProject.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


cv::Mat applyTrafo2(cv::Mat& img, cv::Mat T)
{

    //Here is a better and faster implemented way of doing the affine transformation
    cv::Mat newImg;
    cv::warpPerspective(img, newImg, T, img.size());
    return newImg;

}
    
cv::Mat createTrafo(float angle, float tx, float ty, float sx, float sy, float skew)
{
    //Here we are going to create Transformations - Rotation, Translation, Scaling factor and skew
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F); //We are going to create a Rotation matrix
    
    R.at<float>(0, 0) = cos(angle);
    R.at<float>(0, 1) = -sin(angle);
    R.at<float>(1, 0) = sin(angle);
    R.at<float>(1, 1) = cos(angle);


    cv::Mat t = cv::Mat::eye(3, 3, CV_32F); //Translational matrix

    t.at<float>(0, 2) = tx;
    t.at<float>(1, 2) = ty;

    cv::Mat S = cv::Mat::eye(3, 3, CV_32F); //Scaling

    S.at<float>(0, 0) = sx;
    S.at<float>(1, 1) = sy;


    cv::Mat Skew = cv::Mat::eye(3, 3, CV_32F); //Skewwing

    Skew.at<float>(0, 1) = skew;
    

    return t * R * Skew * S; //We multiple them together to get the full Transformation. And here the order of the Multiplication Matter

}

cv::Mat applyTrafo(cv::Mat& img, cv::Mat T) // That '&' means that we get a reference, so we don't copy the whole/original image
{
    //Here we are going to apply those Transformatioam
    cv::Mat newImg = cv::Mat::zeros(img.size(), CV_8UC3); //CV_8UC3 means an RGB image
    
    //We can get the height and row of the image
    int HEIGHT = img.rows;
    int WIDTH = img.cols;

    for (int i = 0; i < HEIGHT; ++i) //We go along the Height of the image
    {
        for (int j = 0; j < WIDTH; ++j) //We go long the Width of the image - We do the transformation on every pixel
        {
            cv::Mat p(3, 1, CV_32F); //We need to have a Homogenious vector on the pixel coordinates - We need a column vector with 3 elements. Last element should always be 1.
            p.at<float>(0, 0) = i;
            p.at<float>(1, 0) = j;
            p.at<float>(2, 0) = 1.f;

            cv::Mat newP = T * p; //We get the new point coordinates from multiplying the old point with the Transformation matrix. That is why we need a 3x1 column vector (3 row, 1 column) because the T matrix is a 3x3 matrix, and we need it for the matrix multiplication
            int newX = newP.at<float>(0, 0);
            int newY = newP.at<float>(1, 0);

            //This if ensures that we will never be out of bounds. Otherwise we will leave it Black
            if (0 <= newX && newX < HEIGHT && 0 <= newY && newY < WIDTH)
            {
                //This line means that the new x and y coordinates should be the same color as the original image
                newImg.at<cv::Vec3b>(newX, newY) = img.at<cv::Vec3b>(i, j); //Is a matrix whose elements are RGB value, so its elements are vectors, with 3 elements that are bytes
            }
        }

    }

    return newImg; //We get back the new image

}


int main()
{
    cv::Mat A = cv::Mat::zeros(3,3, CV_64F); //MAtrix with double precision, here we create a 3x3 zeros matrix
    cv::Mat B(3, 1, CV_64F); //Column vector

    B.at<double>(0, 0) = 1; //Here we tell to the B vector, that we want to assign float value, to the 0,0 place. Row first, Column Second
    B.at<double>(1, 0) = 2;
    B.at<double>(2, 0) = 3;
       

    //We can print them out, and also we can make matrix operations with them
    std::cout << "A: " << A << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "A * B: " << A*B << std::endl;


    //Make the Transformation

    cv::Mat img = cv::imread("T:\\opencv\\sources\\samples\\data\\apple.jpg");

    //Define the base parameters
    float angle = 0.5;
    float tx = 1;
    float ty = 1;
    float sx = 1;
    float sy = 1;
    float skew = 1;


    while (true) {
        cv::Mat T = createTrafo(angle, tx, ty, sx, sy, skew); //The first value, the angle is in radians
        cv::Mat newImg = applyTrafo2(img, T);


        cv::imshow("Affine Transformation", newImg);
        int key = cv::waitKey(0);

        if (key == 'q') {
            break;
        }
        else if (key == 'c') { //Press c to change the parameters
            std::cout << "Please enter the angle: ";
            std::cin >> angle;
            std::cout << "Please enter the tx: ";
            std::cin >> tx;
            std::cout << "Please enter the ty: ";
            std::cin >> ty;
            std::cout << "Please enter the sx: ";
            std::cin >> sx;
            std::cout << "Please enter the sy: ";
            std::cin >> sy;
            std::cout << "Please enter the skew: ";
            std::cin >> skew;
        }
    }
    cv::destroyAllWindows();
    
    



    return 0;
}


