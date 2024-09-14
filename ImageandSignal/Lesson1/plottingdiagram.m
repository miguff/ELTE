

%Here we are going to plot a lot of functions: parabola, exponentioal,
%sine, cosine, sinc

%1. Parabola
x = linspace(-2, 2, 101); %It creates a neumber list between -2 and 2 divided into 101 different number
y = x.^2; %We put it to the square
figure
plot(x, y)
title("parabola")

%2. Parabola
x = linspace(-10, 2, 101);
y = exp(x); %It creates the exponential function
figure
plot(x, y, 'g', 'LineWidth', 2) %We change something in the plot style, make it green
title("Exponential")


%3. sine, cosine

x = linspace(-2*pi, 2*pi, 101); %It creates a number list between -2*pi and 2*pi divided into 101 different number
figure
plot(x, sin(x), 'b', x, cos(x), "r--") %We can specify multiple plots in one plot
legend("Sine", "Cosine") %It gives what each line means
axis tight


%4. sinc function

x = linspace(-4, 4, 101);
y = sin(pi*x)./(pi*x); %Here is how we can calculate the sinc function
y((end+1)/2) = 1; %Here we make the middle value 1, just to be precise
figure
plot(x, y)

%But for all this, we can use the MATLAB sinc function
%Somehow this does not seem to work
%This somehow does not work in MATLAB 2023b
%y = sinc(x);
%figure
%plot(x,y)

