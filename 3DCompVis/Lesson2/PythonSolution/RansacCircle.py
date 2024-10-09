import matplotlib.pyplot as plt
import numpy as np


class GenerateCircle:
    def __init__(self, numberofpoints, numofoutliers, radius, outlierxrange = (-30, 30), outlieryrange = (-30, 30), xnormal = (0,2), ynormal = (0,2)):
        self.numberofpoints = numberofpoints
        self.numberofoutliers = numofoutliers
        self.radius = radius
        self.outlierxrange = outlierxrange
        self.outlieryrange = outlieryrange
        self.xnormal = xnormal
        self.ynormal = ynormal
    
    def circlegen(self):
        # Generate circular distribution points
        angles = np.linspace(0, 2 * np.pi, self.numberofpoints)
        x_circle = self.radius * np.cos(angles) + np.random.normal(self.xnormal[0], self.xnormal[1], self.numberofpoints)
        y_circle = self.radius * np.sin(angles) + np.random.normal(self.ynormal[0], self.ynormal[1], self.numberofpoints)

        # Generate random outliers
        x_outliers = np.random.uniform(self.outlierxrange[0], self.outlierxrange[1], self.numberofoutliers)
        y_outliers = np.random.uniform(self.outlieryrange[0], self.outlieryrange[1], self.numberofoutliers)

        x = np.append(x_circle,x_outliers)
        y = np.append(y_circle,y_outliers)
        return x, y
    

class RANSAC:
    def __init__(self, Points):
        self.Points = Points


    

    def fitCircle(self):

        x1, y1 = self.Points[0]
        x2, y2 = self.Points[60]
        x3, y3 = self.Points[120]


        # Set up the matrices
        A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])
        B = np.array([
        [x1**2 + y1**2],
        [x2**2 + y2**2],
        [x3**2 + y3**2]
    ])

    # Solve the system of equations A * [h, k, r] = B
    # We need to find the values of h, k, and r
    # We can compute the pseudo-inverse since A is not square
        A_inv = np.linalg.pinv(A)
        X = A_inv @ B

        h = X[0, 0] / 2
        k = X[1, 0] / 2
        r = np.sqrt(h**2 + k**2 + X[2, 0])  # Calculate radius

        return h, k, r


def main():
    Cirle = GenerateCircle(150, 40,20)
    cirlex, Cirley = Cirle.circlegen()

    Points = np.column_stack((cirlex, Cirley))
    
    Ransacgen = RANSAC(Points)
    h,k,r = Ransacgen.fitCircle()

    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Parametric equations for the circle
    x_circle = h + r * np.cos(theta)
    y_circle = k + r * np.sin(theta)

    # Create a plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_circle, y_circle, label='Fitted Circle', color='blue')
    plt.scatter(h, k, color='red', label='Center (h, k)')
    plt.scatter(cirlex, Cirley, color='green', label='Given Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Circle Fitting to Points')
    plt.axis('equal')  # Equal scaling
    plt.legend()
    plt.show()


"""
Hogyan fogom megoldani.

1. Be kell adni az összes pontot a RANSAC classnak
2. A RANSAC class kiválaszt 3 pontot és meghatározza belőlük a kört
3. Ezután lesz az ellenőrzés funkció, hogy megnézi, hogy az adott kör + határértéken belül mennyi elem van, számol egy MSE-t.
4. Ezután összehasonlítja, hogy ez a modell volt-e a jobb vagy az előző. Amelyiknek kevesebb lesz a MSE-je azt viszi tovább
5.Majd ezt addig csinálja amíg végig nem fut a megadott mennyiség számon

"""


if __name__ == "__main__":
    main()