%Here we are going to implement the bisection method to solve the equation
%sin(x) = log(x) and we will plot the results



f = @(x) (sin(x) - log(x)); % function handle
x = bisection(f, 1, 3, 1e-4); % bisection solution
t = linspace(0, 2*pi); % uniform sampling
hold on
plot(t, sin(t), 'b') % sine curve
plot(t, log(t), 'r') % log curve
plot(x, sin(x), 'ko') % intersection
rectangle('Position', [1, 0, 2, log(3)]) % area of interest
grid on




% BISECTION method to solve f(x) = 0 for x
%
% Parameters:
% f function handle
% a, b endpoints of interval
% (assuming b > a and f(a)*f(b) < 0
% eps error tolerance
% Returns:
% x solution for f(x) = 0
%
function x = bisection(f, a, b, eps)
    ya = f(a);
    yb = f(b);
    err = (b - a) / 2;
    while err > eps
        x = (a + b) / 2;
        y = f(x);
        if y * ya > 0
            a = x;
        elseif y * yb > 0
            b = x;
        else
            break;
        end
        err = err / 2;
    end
    x = (a + b) / 2;
end
