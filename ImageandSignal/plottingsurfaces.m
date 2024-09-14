%Here we are going to plot some surfaces: Paraboloid, Gaussian function,
%sphere, torus


%Paraboloid over [-2,2]x[-2,2]
x = linspace(-2,2,20);
y = linspace(-2,2, 20);

[X, Y] = meshgrid(x, y); %returns 2-D grid coordinates based o the coordinates contained in vectors x and y. X is a matrix where each row is a copy of x, and  Y is a matrix where each column is a copy of y. the grid representes by the coordinates X and Y ha lenght(y) rows and lengt(x) xolumns
Z = X.^2 + Y.^2;
figure
surf(X, Y, Z)
shading interp
title("Paraboloid")


%Gaussian function over [−4, 4] × [−4, 4]:
x = linspace(-4,4,101);
y = linspace(-4,4, 101);
[X, Y] = meshgrid(x, y);
Z = exp(-(X.^2 + Y.^2) / 2);
figure
surf(X, Y, Z) 
shading interp
title("Gaussian")


%Sphere, using parametric form x = cos(u)*sin(v), y = sin(u)*sin(v), z = cos(v) (u ∈ [0, 2π], v ∈ [0, π]):
u = linspace(0, 2*pi, 33); % uniform sampling
v = linspace(0, pi, 17);
[U, V] = meshgrid(u, v);
X = cos(U).*sin(V);
Y = sin(U).*sin(V);
Z = cos(V);
figure
surf(X,Y,Z)
title("Sphere")


%Torus, using parametric form x = (R + r*cos(u))*cos(v), x = (R + r*cos(u))*sin(v), z = r*sin(u) (u ∈ [0, 2π], v ∈ [0, 2π]):
R = 1; r = 0.25;
u = linspace(0, 2*pi, 17);
v = linspace(0, 2*pi, 33);

[U, V] = meshgrid(u, v);

X = (R + r*cos(U)).*cos(V); %Here these are element-wise multiplications, which means that each element in (R + r*cos(U)) is multiplied by the corresponding element in cos(V)
Y = (R + r*cos(U)).*sin(V);
Z = r*sin(U);

figure
surf(X,Y,Z)
title("Torus")

%I have to check the multiplication in matlab, which is elemnt wise, which
%is matrix multiplication, etc, etc







