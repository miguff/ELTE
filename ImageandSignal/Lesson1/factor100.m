%How to calculate the 100! for multiple ways

%1. with a foor loop

s = 1;
for i = 2:100;
    s = i*s;
end

display(s)

%2. we can the 'prod' or the 'fact' functions
k = prod(1:100); %gives back the product elment of an array
z = prod(5);

matrix = [[2 2]; [4 4]];
matrix

matrixprod = prod(matrix,2) %Here it sums either the rows or the columns, based on the dimension so set.


disp(k)
disp(z)

%Factorial gives back, the factorial that we want
l = factorial(100);
o = factorial(5);

disp(l)
disp(o)