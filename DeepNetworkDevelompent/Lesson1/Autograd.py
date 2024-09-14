import numpy as np
import torch
import matplotlib.pyplot as plt


def main():

    #set requires_grad=true if you need the grad later
    x = torch.tensor(5., requires_grad=True)
    print(x)

    y = 3*x**2+4*x+2
    print(y)

    #The backward() method is used to compute the gradient during the backward pass in a neural network
    y.backward() #--> this is required if we wanted to calculate the x.grad, which is the derivative of y with respect to the variable x.

    #Here it calculates the derivate
    x.grad
    print(x.grad)

    #We can calculate multiple derivates of a function with a foor loop. but every .backward után kell egy .grad
    multiplederivates()



def multiplederivates():
    x = torch.tensor(5., requires_grad=True)
    for i in range(3):
        y = 3*x**2 + 4*x + 2
        y.backward()
        print(x.grad)
    #x.grad.zero_() ## pytorch accumulates the derivatives by default


if __name__ == "__main__":
    main()