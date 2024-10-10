import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import math

import numpy as np

class GenerateCircle:
    """
    A class to generate a set of points distributed in a circular pattern
    with some outlier points.

    Attributes:
        numberofpoints (int): The number of points to generate in a circular distribution.
        numberofoutliers (int): The number of outlier points to generate.
        radius (float): The radius of the circle.
        outlierxrange (tuple): The range for the x-coordinates of outliers (min, max).
        outlieryrange (tuple): The range for the y-coordinates of outliers (min, max).
        xnormal (tuple): Uniformly distributed floats for the normal distribution in x.
        ynormal (tuple): Uniformly distributed floats for the normal distribution in y.
    """
    
    def __init__(self, numberofpoints, numofoutliers, radius,
                 outlierxrange=(-30, 30), outlieryrange=(-30, 30),
                 xnormal=(0, 2), ynormal=(0, 2)):
        """
        Initializes the GenerateCircle instance with specified parameters.

        Parameters:
            numberofpoints (int): Number of points to generate in the circle.
            numofoutliers (int): Number of outlier points to generate.
            radius (float): Radius of the circle.
            outlierxrange (tuple): Range for the x-coordinates of outliers (min, max). Default is (-30, 30).
            outlieryrange (tuple): Range for the y-coordinates of outliers (min, max). Default is (-30, 30).
            xnormal (tuple): Uniformly distributed floats over x-coordinates of normal points. Default is (0, 2).
            ynormal (tuple): Uniformly distributed floats over y-coordinates of normal points. Default is (0, 2).
        """
        self.numberofpoints = numberofpoints
        self.numberofoutliers = numofoutliers
        self.radius = radius
        self.outlierxrange = outlierxrange
        self.outlieryrange = outlieryrange
        self.xnormal = xnormal
        self.ynormal = ynormal

    def circlegen(self):
        """
        Generates circular distribution points along with random outliers.

        Returns:
            tuple: Two numpy arrays containing the x-coordinates and y-coordinates
                   of the generated points (circular points and outliers).
        """
        # Generate circular distribution points
        angles = np.linspace(0, 2 * np.pi, self.numberofpoints)
        x_circle = self.radius * np.cos(angles) + np.random.normal(self.xnormal[0], self.xnormal[1], self.numberofpoints)
        y_circle = self.radius * np.sin(angles) + np.random.normal(self.ynormal[0], self.ynormal[1], self.numberofpoints)

        # Generate random outliers
        x_outliers = np.random.uniform(self.outlierxrange[0], self.outlierxrange[1], self.numberofoutliers)
        y_outliers = np.random.uniform(self.outlieryrange[0], self.outlieryrange[1], self.numberofoutliers)

        x = np.append(x_circle, x_outliers)
        y = np.append(y_circle, y_outliers)
        return x, y

import numpy as np
import pandas as pd
import random
import math

class RANSAC:
    """
    A class that implements the RANSAC (Random Sample Consensus) algorithm
    to fit a circle to a set of points, allowing for the presence of outliers.

    Attributes:
        Points (array-like): The input points to fit the circle.
        numberofiter (int): The number of iterations for the RANSAC algorithm.
        h (float): x-coordinate of the circle center.
        k (float): y-coordinate of the circle center.
        r (float): Radius of the circle.
        insidepoints (int): Count of inlier points inside the parameter circle.
        parameter (float): The parameter to define the inner and outer radius.
        error_model (int): The error model used (0 for point count, 1 for distance error).
        error (float): The current error measure for the best fit.
    """

    def __init__(self, Points, numberofiter, parameter=5, error_model=0):
        """
        Initializes the RANSAC instance with specified parameters.

        Parameters:
            Points (array-like): The points to fit the circle.
            numberofiter (int): Number of iterations for the RANSAC algorithm.
            parameter (float): Parameter for defining inliers (default is 5).
            error_model (int): The error model (0 for count, 1 for distance error; default is 0).
        """
        self.Points = Points
        self.numberofiter = numberofiter
        self.h = None
        self.k = None
        self.r = None
        self.insidepoints = 0
        self.parameter = parameter
        self.error_model = error_model
        self.error = 99999

    def fit(self):
        """
        Fits a circle to the input points using the RANSAC algorithm.

        Returns:
            tuple: The (h, k, r) parameters of the fitted circle.
        """
        for i in range(self.numberofiter):
            print(f"{i + 1} iteration")
            newh, newk, newr = self.fitCircle()
            if self.error_model == 0:
                newinsidepoints = self.Error(newh, newk, newr)
                print(f"New inside points: {newinsidepoints} \nOld inside points: {self.insidepoints}")
                if newinsidepoints > self.insidepoints:
                    self.insidepoints = newinsidepoints
                    self.h = newh
                    self.k = newk
                    self.r = newr
            else:
                PointsDf = pd.DataFrame(self.Points, columns=["datax", "datay"])
                PointsDf["DistanceError"] = PointsDf.apply(lambda row: math.sqrt((row['datax'] - newh) ** 2 + (row['datay'] - newk) ** 2), axis=1)
                PointsDf["DistanceError"] = PointsDf["DistanceError"] - newr
                error = TSE(PointsDf)
                print(f"New error points: {error} \nOld error points: {self.error}")
                if error < self.error:
                    self.error = error
                    self.h = newh
                    self.k = newk
                    self.r = newr

        return self.h, self.k, self.r

    def Error(self, h, k, r):
        """
        Calculates the number of inlier points within the defined radius.

        Parameters:
            h (float): x-coordinate of the circle center.
            k (float): y-coordinate of the circle center.
            r (float): Radius of the circle.

        Returns:
            int: The count of points that lie within the specified radius range.
        """
        greaterR = r + self.parameter
        lesserR = r - self.parameter
        Pointsdf = pd.DataFrame(self.Points, columns=["datax", "datay"])
        Pointsdf["Inside Point"] = Pointsdf.apply(lambda row: (row['datax'] - h) ** 2 + (row['datay'] - k) ** 2 < greaterR ** 2 and
                                                               (row['datax'] - h) ** 2 + (row['datay'] - k) ** 2 > lesserR ** 2, axis=1)
        insidepoints = Pointsdf["Inside Point"].sum()
        return insidepoints

    def fitCircle(self):
        """
        Randomly selects three points from the dataset to fit a circle.

        Returns:
            tuple: The (h, k, r) parameters of the fitted circle.
        """
        random1, random2, random3 = random.sample(range(0, len(self.Points)), 3)
        x1, y1 = self.Points[random1]
        x2, y2 = self.Points[random2]
        x3, y3 = self.Points[random3]

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
        A_inv = np.linalg.pinv(A)
        X = A_inv @ B

        h = X[0, 0] / 2
        k = X[1, 0] / 2
        r = np.sqrt(h**2 + k**2 + X[2, 0])  # Calculate radius

        return h, k, r

def TSE(DistanceError: pd.DataFrame):
    """
    Computes the Total Squared Error (TSE) for the distance errors in the DataFrame.

    Parameters:
        DistanceError (pd.DataFrame): DataFrame containing distance errors.

    Returns:
        float: The Mean Squared Error (MSE) calculated from TSE.
    """
    DistanceError["DistanceError"] = DistanceError["DistanceError"] ** 2
    TSE = DistanceError["DistanceError"].sum()
    return MSE(TSE, len(DistanceError))

def MSE(TSE, numberofPoints):
    """
    Computes the Mean Squared Error (MSE) from the Total Squared Error (TSE).

    Parameters:
        TSE (float): Total Squared Error.
        numberofPoints (int): Number of points used to compute MSE.

    Returns:
        float: The Root Mean Squared Error (RMSE) calculated from MSE.
    """
    MSE = TSE / numberofPoints
    return RMSE(MSE)

def RMSE(MSE: int):
    """
    Computes the Root Mean Squared Error (RMSE) from the Mean Squared Error (MSE).

    Parameters:
        MSE (int): Mean Squared Error.

    Returns:
        float: The calculated RMSE.
    """
    RMSE = math.sqrt(MSE)
    return RMSE


def main():
    Cirle = GenerateCircle(400, 200,20, xnormal=(0, 2), ynormal=(0, 2))
    cirlex, Cirley = Cirle.circlegen()

    Points = np.column_stack((cirlex, Cirley))
    
    Ransacgen = RANSAC(Points, 1000, parameter=1.5, error_model=0)
    h,k,r = Ransacgen.fit()

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


if __name__ == "__main__":
    main()