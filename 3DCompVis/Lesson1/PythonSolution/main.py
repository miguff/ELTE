import cv2
import numpy as np
import sys


def main():
    
    ##################### Widow Setup #########################
    BGRowL: int = 400 
    BGColL: int = 400
    BorderSize: int = 10
    #Text setuo
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    ################### Platform Setup ##########################
    PlatformWidth = 100
    PlatformHeight = 10


    PlatformPos = np.array([BGRowL // 2 - PlatformWidth // 2, BGColL - 20])
    PlatformVelocity = 6


    ################# BALL SETUP ###########################
    #Ball start point and Radius setup
    CircleCx: int = 100 #Starting x coordinate of X
    CircleCy: int = 100 #Starting y coordinate of x
    CircleR: int = 20 #Circle Radius

    #Ball speed Setup
    dx: int = 5 #Speed of x
    dy: int = 1 #Speed of y
    BallPos = np.array([CircleCx, CircleCy])
    BallSpeed = np.array([dx, dy])

    point: int = 0
    while True:
        Background = np.zeros((BGRowL,BGColL,3), np.uint8)
        cv2.rectangle(Background, (0,0), (BorderSize,BGRowL),(255,255,255), -1)
        cv2.rectangle(Background, (0,0), (BGColL,BorderSize),(255,255,255), -1)
        cv2.rectangle(Background, (BGColL-BorderSize, 0), (BGColL,BGRowL),(255,255,255), -1)
        cv2.putText(Background, f"Here is your point {point}", (20,40), font, fontScale, color, 1, cv2.LINE_AA)
        BallPos += BallSpeed

        if cv2.waitKey(1) == ord('a') and PlatformPos[0] > BorderSize:
            PlatformPos[0] -= PlatformVelocity

        elif cv2.waitKey(1) == ord('d') and PlatformPos[0]+PlatformWidth < BGColL-BorderSize:
            PlatformPos[0] += PlatformVelocity
       

        ############## Collision Detection ########################
        #Collision with Left and Right Wall
        if BallPos[0]+CircleR > BGColL-BorderSize or BallPos[0]-CircleR < BorderSize:
            BallSpeed[0] *= -1

        #Collision with Top Wall    
        if BallPos[1]-CircleR < BorderSize:
            BallSpeed[1] *= -1

        #Falling down
        if BallPos[1] > BGRowL or BallPos[0] > BGColL or BallPos[1] < 0 or BallPos[0] < 0:
            cv2.putText(Background, "Game OVER", ((BGRowL // 2)-50, BGColL // 2), font, fontScale, color, thickness, cv2.LINE_AA)
            

        ###################### Collision with the platform
        if (PlatformPos[1] <= BallPos[1] + CircleR <= PlatformPos[1] + PlatformHeight and
            PlatformPos[0] <= BallPos[0] <= PlatformPos[0] + PlatformWidth):
            BallSpeed[1] *= -1
            point += 1


        ######################### Create the objects ###########################x
        #Make the circle
        cv2.circle(Background, tuple(BallPos), CircleR, (255,255,255), -1)
        
        #Make the Platforms
        cv2.rectangle(Background, tuple(PlatformPos),
                  (PlatformPos[0] + PlatformWidth, PlatformPos[1] + PlatformHeight),
                  (255,0,255), -1)
    
        cv2.imshow('Lesson 1 - Bounce Game - Python',Background)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        








if __name__ == "__main__":
    main()