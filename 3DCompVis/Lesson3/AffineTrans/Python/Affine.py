import numpy as np
import cv2
import tkinter as tk


# Global transformation parameters
angle = 0.0
tx = 0.0
ty = 0.0
sx = 1.0
sy = 1.0
skew = 0.0


def main():
    
    img = cv2.imread("T:\\opencv\\sources\\samples\\data\\apple.jpg")
    global angle, tx, ty, sx, sy, skew

    while True:

        TMatrix = createTrafo(angle ,tx, ty, sx, sy, skew)
        newImg = ApplyTrafo(img, TMatrix)

        # Parameters for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (5, 50)
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
 
 
        newImg = cv2.putText(newImg, '"c" to change the parameters', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)


        cv2.imshow("Affine Transformation", newImg)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('c'): #Change the Parameters
            OpenGUI()

    cv2.destroyAllWindows()


def OpenGUI():
    # We Create a new Tinker window for our values
    root = tk.Tk()
    root.geometry("400x300")
    root.title("Change Transformation Parameters")

    # We declare the variables to be changes
    angle_var = tk.StringVar(value=str(angle))
    tx_var = tk.StringVar(value=str(tx))
    ty_var = tk.StringVar(value=str(ty))
    sx_var = tk.StringVar(value=str(sx))
    sy_var = tk.StringVar(value=str(sy))
    skew_var = tk.StringVar(value=str(skew))

    # Function to handle submission
    def submit():
        global angle, tx, ty, sx, sy, skew
        try:
            angle = float(angle_var.get())
            tx = float(tx_var.get())
            ty = float(ty_var.get())
            sx = float(sx_var.get())
            sy = float(sy_var.get())
            skew = float(skew_var.get())
        except ValueError:
            print("Invalid input. Please enter valid float values.")
        root.destroy()

    # Create input fields and labels
    tk.Label(root, text='Rotation Angle (radians):').grid(row=0, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=angle_var).grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text='Translation X:').grid(row=1, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=tx_var).grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text='Translation Y:').grid(row=2, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=ty_var).grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text='Scaling X:').grid(row=3, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=sx_var).grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text='Scaling Y:').grid(row=4, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=sy_var).grid(row=4, column=1, padx=10, pady=5)

    tk.Label(root, text='Skew:').grid(row=5, column=0, padx=10, pady=5, sticky='e')
    tk.Entry(root, textvariable=skew_var).grid(row=5, column=1, padx=10, pady=5)


    tk.Button(root, text='Submit', command=submit).grid(row=6, column=0, columnspan=2, pady=20)


    root.mainloop()



def createTrafo(angle: float, tx: float, ty: float, sx:float, sy:float, skew:float):

    RotationMatrix = np.eye(3,3)
    RotationMatrix[0, 0] = np.cos(angle)
    RotationMatrix[0, 1] = -np.sin(angle)
    RotationMatrix[1, 0] = np.sin(angle)
    RotationMatrix[1, 1] = np.cos(angle)

    TranslationMatrix = np.eye(3,3)
    TranslationMatrix[0, 2] = tx
    TranslationMatrix[1,2]= ty

    ScalingMatrix = np.eye(3,3)
    ScalingMatrix[0,0] = sx
    ScalingMatrix[1,1] = sy

    SkewMatrix = np.eye(3,3)
    SkewMatrix[0, 1] = skew

    return TranslationMatrix @ RotationMatrix @ SkewMatrix @ ScalingMatrix


def ApplyTrafo(Image: np.ndarray, TransMatrix: np.ndarray):
    ImageCopy = Image.copy()

    affine_matrix = TransMatrix[:2, :3]

    newImg = np.zeros(ImageCopy.shape, dtype=ImageCopy.dtype)
    HEIGHT, WIDTH, Channel = ImageCopy.shape
    

    newImg = cv2.warpAffine(ImageCopy, affine_matrix, (WIDTH, HEIGHT))


    #In python we do not the walk over every row and column manually, it has a built in function fo rie
    # for i in range(HEIGHT):
    #     for j in range(WIDTH):
    #         p = np.ndarray((3,1))
    #         p[0, 0] = i
    #         p[1,0] = j
    #         p[2, 0] = 1

    #         newP = TransMatrix @ p
    #         newX = int(newP[0,0])
    #         newY = int(newP[1,0])

    #         if 0 <= newX and newX < HEIGHT and 0 <= newY and newY < WIDTH:
    #             newImg[newX, newY] = ImageCopy[i, j]

    return newImg

if __name__ == "__main__":
    main()