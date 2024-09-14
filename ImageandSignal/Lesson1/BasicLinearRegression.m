N = 100; %Number of random points
c = [2, 1]; %Coefficients of the original line
X = rand(N, 1);  %Points generation
y = c(2) + c(1)*X + randn(N, 1);
A = [X, ones(N, 1)]; %This is the Vandermonde Matrix - never heard of is

%We can get this cf in multiple ways
%using pinv(A)
cf = pinv(A) * y; %returns the Moore-Penrose Pseudoinverse of matrix A.

%Using MATLAB mldivide function
cf = A\y;

%Using MATLAB built-in polynomial fitting function
cf = polyfit(X, y, 1);



hold on
plot(X, y, 'b.')
plot([0 1], c(2) + c(1) * [0 1], 'k--')
plot([0 1], cf(2) + cf(1) * [0 1], 'r')
legend('Points', 'Original line', 'Fitted line')
grid on

