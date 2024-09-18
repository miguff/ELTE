import cv2
import numpy as np



def main():
    
    BGRowL: int = 400
    BGColL: int = 400
    BorderSize: int = 10

    dx: int = 5 #Speed of x
    dy: int = -1 #Speed of y

    CircleCx: int = 100 #Starting x coordinate of X
    CircleCy: int = 100 #Starting y coordinate of x
    CircleR: int = 20 #Circle Radius

    PlatformXL = 300
    PlatformXR = 700
    SpeedofPlatform = 10

    Background = np.zeros((BGRowL,BGColL,3), np.uint8)

    while True:
        cv2.imshow('image',Background)
        Background = np.zeros((BGRowL,BGColL,3), np.uint8)
        cv2.rectangle(Background, (0,0), (BorderSize,BGRowL),(255,255,255), -1)
        cv2.rectangle(Background, (0,0), (BGColL,BorderSize),(255,255,255), -1)
        cv2.rectangle(Background, (BGColL-BorderSize, 0), (BGColL,BGRowL),(255,255,255), -1)
        
        CircleCx = CircleCx+dx
        CircleCy = CircleCy+dy

        if cv2.waitKey(1) == ord('a'):
            PlatformXL = PlatformXL-SpeedofPlatform
            PlatformXR = PlatformXR -SpeedofPlatform
        elif cv2.waitKey(1) == ord('d'):
            PlatformXL = PlatformXL+SpeedofPlatform
            PlatformXR = PlatformXR +SpeedofPlatform
        

        if CircleCx+CircleR > BGColL-BorderSize or CircleCx-CircleR < BorderSize:
            dx = -dx
        if CircleCy-CircleR < BorderSize:
            dy = -dy


        if CircleCy > BGRowL or CircleCx > BGColL or CircleCy < 0 or CircleCx < 0:
            break

        cv2.circle(Background, (CircleCx, CircleCy), CircleR, (255,255,255), -1)
        

        cv2.rectangle(Background, (PlatformXL, BGRowL-2*BorderSize), (PlatformXR, BGRowL-BorderSize), (255,255,255), -1)
        cv2.imshow('image',Background)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        








if __name__ == "__main__":
    main()