
N = 10; % matrix size
M = 10; % number of iterations
A = randi(10, N, N); % NxN random integer matrix
A = (A + A') / 2; % symmetric transform
x = 2 * rand(N, 1) - 1; % random starting point
x = x / norm(x, 2); % normalization
lambda = zeros(1, N); % eigenvalue estimates
for i = 1:M
x0 = x;
x = A * x; % power iteration step
lambda(i) = x' * x0; % Rayleigh quotient
x = x / norm(x, 2); % normalization
end
lambda0 = max(abs(eig(A))); % MATLAB eigenvalue estimate
hold on
plot(lambda, 'b')
plot(1:M, lambda0 * ones(1, M), 'r--')