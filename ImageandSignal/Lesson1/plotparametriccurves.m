
%Here we are going to plot the graph of the parametric curves: cirle,
%spirals, helix



%Unit circle, using the parametric form x = cos(t), y = sin(t) (t is
%element os [0, 2*pi])
t = linspace(0, 2*pi, 101);
figure
plot(cos(t), sin(t))
axis equal
disp(cos(0))
disp(sin(0))



%Archimedean sprilar, using the parametric form: x = (a+ b*t)*cos(t), y =
%(a+b*t)*sin(t)
t = linspace(0, 8*pi, 1001);
figure
%We specify the parameters
a = 1;
b = 2;
plot((a+b*t).*cos(t), (a+b*t).*sin(t))
axis equal

%Logarithmic spiral, using the parametric form x = a*e^(b-t)*cos(t), y =
%a*e^(b-t)*sin(t) (t elemen of R)

t = linspace(0, 8*pi, 1001);
figure
%We specify the parameters
a = 1;
b = 0.2;
plot(a*exp(b*t).*cos(t), a*exp(b*t).*sin(t))
axis equal

%Helix, using, the parametrix for as a*cos(t), y = a*sin(t), z = bt (t>=0)
t = linspace(0, 2*pi, 101);
figure
plot(cos(t), sin(t))
axis equal