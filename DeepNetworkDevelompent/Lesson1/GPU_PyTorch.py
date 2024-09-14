import numpy as np
import matplotlib.pyplot as plt # for plotting
# install it from: https://pytorch.org/get-started/locally/
import torch # the most used deep learning framework next to TensorFlow
import time # measure time


def main():

    numpyspeed()
    pytorchspeed()

    #Pythorc has some of the same applications as numpy but much faster like
    """
    - numpy.ndarray vs torch.tensor
    - numpy.ones vs torch.ones
    - numpy.zeros vs torch.zeros
    - numpy.copy() vs torch.clone()
    - .reshape vs .view (numpy vs torch)
    - numpy.linalg.inv() vs torch.inverse()
    - torch.from_numpy()
    - torch.where()
    """

    MatrixInversDetTrans()


def numpyspeed():
    #Here, we will see that, Pytorch is faster
    matrix_1 = np.random.randn(5000,5000)
    matrix_2 = np.random.randn(5000,5000)

    tstart = time.time()
    res_numpy = np.dot(matrix_1, matrix_2)
    tend = time.time()
    print('Numpy calculation time pytorch: ',tend-tstart, ' s')
    print(res_numpy[0])

def pytorchspeed():
    matrix_1 = np.random.randn(5000,5000)
    matrix_2 = np.random.randn(5000,5000)
    torch_mx_1 = torch.from_numpy(matrix_1).float().cuda()
    torch_mx_2 = torch.from_numpy(matrix_2).float().cuda()  

    tstart = time.time()
    res_pytorch = torch.mm(torch_mx_1, torch_mx_2)
    tend = time.time()
    print('Pytorch calculation time pytorch: ',tend-tstart, ' s')
    print(res_pytorch[0])

def MatrixInversDetTrans():
    n = np.array([[2.,3.],[5.,5.]], dtype=np.float32)
    p = torch.tensor([[2.,3.],[5.,5.]], dtype=torch.float32, device='cuda:0')

    print(f'numpy inverse:\n {np.linalg.inv(n)}')
    print(f'torch inverse:\n {p.inverse()}')
    print()

    print(f'numpy determinant:\n {np.linalg.det(n)}')
    print(f'torch determinant:\n {torch.det(p)}')
    print()

    print(f'numpy transpose:\n {n.T}')
    print(f'torch transpose:\n {p.t()}')
    print()


if __name__ == "__main__":
    main()