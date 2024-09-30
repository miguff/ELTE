#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream> 

#define WIDTH 400
#define HEIGHT 400
#define BORDERSIZE 10

cv::Mat Background;

int Circlex, Circley, CircleR;
int dx, dy;
int PlatformW, PlatformH, PlatformV;
int point;

int main()
{
    

    //################# Platform Setup #####################
    PlatformW = 100;
    PlatformH = 10;
    PlatformV = 6;

    int Platformpos[2] = {WIDTH / 2 - PlatformW / 2, HEIGHT - 20};


    //##################### BALL Setup ######################x
    //Start point and Radius setup
    Circlex = 100;
    Circley = 100;
    CircleR = 20;

    //Ball speed setup
    dx = 15;
    dy = 5;


    
    int BallPos[2] = { Circlex, Circley };
    int BallSpeed[2] = { dx, dy };
    point = 0;

    while (true) {
        Background = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

        //Left Side Wall
        cv::rectangle(Background, cv::Point(0, 0), cv::Point(BORDERSIZE, HEIGHT), cv::Scalar(255, 255, 255), cv::FILLED);
        //Upper Wall
        cv::rectangle(Background, cv::Point(0, 0), cv::Point(WIDTH, BORDERSIZE), cv::Scalar(255, 255, 255), cv::FILLED);
        //Right Wall
        cv::rectangle(Background, cv::Point(WIDTH-BORDERSIZE, 0), cv::Point(WIDTH, HEIGHT), cv::Scalar(255, 255, 255), cv::FILLED);
        
        //Display the score
        std::stringstream ss;
        ss << "Here is your points: " << point;
        cv::putText(Background, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

        //Setup ball speed
        BallPos[0] +=  BallSpeed[0];
        BallPos[1] +=  BallSpeed[1];

        //Platform movement
        int key = cv::waitKey(30);
        if (key == 'a' && Platformpos[0] > BORDERSIZE) {
            Platformpos[0] -= PlatformV;
        }
        else if (key == 'd' && Platformpos[0] + PlatformW < WIDTH-BORDERSIZE) {
            Platformpos[0] += PlatformV;
        }


        // Collisiions 
        if (BallPos[0] <= BORDERSIZE + CircleR || BallPos[0] >= WIDTH - BORDERSIZE - CircleR) {
            BallSpeed[0] = -BallSpeed[0]; 
        }

        //Top collisoin
        if (BallPos[1] - CircleR <= BORDERSIZE) {
            BallSpeed[1] = -BallSpeed[1]; 
        }
        else if (BallPos[1] >= HEIGHT - CircleR) {
            // Game Over string
            std::stringstream endstring;
            std::stringstream exitString;
            endstring << "Game Over!";
            exitString << "Press 'q' to Exit";
            cv::putText(Background, endstring.str(), cv::Point((HEIGHT /2)  -50, WIDTH / 2), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
            cv::putText(Background, exitString.str(), cv::Point((HEIGHT / 2) -90 , WIDTH / 1.5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
        }
        

        //If if there were a collision with the platform
        if (BallPos[1] + CircleR >= Platformpos[1] && 
            BallPos[1] + CircleR <= Platformpos[1] + PlatformH && 
            BallPos[0] >= Platformpos[0] && 
            BallPos[0] <= Platformpos[0] + PlatformW) { 
            BallSpeed[1] = -BallSpeed[1]; 
            point++;  
        }
        
        
        //Create a ball
        cv::circle(Background, cv::Point(BallPos[0], BallPos[1]), CircleR, cv::Scalar(255, 255, 255), cv::FILLED);


        //Create a platform
        cv::rectangle(Background, cv::Point(Platformpos[0], Platformpos[1]), cv::Point(Platformpos[0] + PlatformW, Platformpos[1] + PlatformH), cv::Scalar(255, 0, 255), cv::FILLED);


        cv::imshow("Lesson 1 - Bounce Game - C++", Background);

        if (cv::waitKey(30) == 'q') {
            break;
        }


    }


    

    return 0;
}
