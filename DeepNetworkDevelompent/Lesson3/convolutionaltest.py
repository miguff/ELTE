import torch
import torch.nn as nn #It creates neural networks


input = torch.tensor([[[0,0,1],
                       [1,2,3],
                       [7,5,3],
                       [5,3,6]],

                  [[9,11,10],
                   [10,11,12],
                   [31,55,32],
                   [17,29,32]]], dtype=torch.float32, requires_grad=True)

print(input)
print(input.shape)


layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
print("------------------MaxPool2d---------------------------------------------")
print(layer(input).shape)
print(layer(input))
print("---------------------------------------------------------------")

layer = nn.AvgPool2d(kernel_size=4, stride=1, padding=2)
print("----------------------AvgPool2D-----------------------------------------")
print(layer(input).shape)
print(layer(input))
print("---------------------------------------------------------------")

layer = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=1, padding='same')
print("-------------------------Conv2D--------------------------------------")
print(layer(input).shape)
print(layer(input))
print("---------------------------------------------------------------")

layer = nn.MaxPool2d(kernel_size = 3, stride = 3)
print("-------------------------MaxPool2D no padding--------------------------------------")
print(layer(input).shape)
print(layer(input))
print("---------------------------------------------------------------")