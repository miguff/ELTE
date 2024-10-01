
%Exercise 4. Simulate and filter recorder malfunction. Take a signal and add salt-pepper noise. Filter it with moving median filter.
%Generation: random samples are affected at given probability (density), and replaced with  white’ or ’black’ values.


fs = 8000; % sampling rate
t = linspace(0, 1, fs + 1); t(end) = []; % sampling points
d = 0.05; % noise density This is 5% I guess
Heartz = 440;
A = 1; % noise intensity
s1 = sin(2*pi * Heartz * t); % musical tone A


%Here we make the indexes where some of the is 1 and other are 0, 
%These are masks. The rand creates values between 0 and 1. Here we check
%that the values are less than 0.5*d (0.5*0.05) if yes, the value will be
%1, if no the value will be 0
test = rand(size(s1))
idx1 = test < 0.5 * d % 'white' indices S
idx2 = rand(size(s1)) < 0.5 * d % 'black' indices



s2 = s1; s2(idx1) = A; s2(idx2) = -A; % noisy signal