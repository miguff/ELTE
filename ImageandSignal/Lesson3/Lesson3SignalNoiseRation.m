%Exercise 3 - Estimate the signal-to-noise ratio (SNR)


fs = 8000; % sampling rate
t = linspace(0, 1, fs + 1); t(end) = []; % sampling points
sigma = 0.25; % noise intensity
s1 = sin(2*pi * 440 * t); % musical tone A
n = sigma * randn(size(s1)); % white noise
s2 = s1 + n; 

%SNR by Psignal/Pnoise, if noise is known:
SNR1 = 10 * log10(norm(s2, 2)^2 / norm(n, 2)^2); % SNR [dB]
SNR2 = 20 * log10(norm(s2, 2) / norm(n, 2)); % reformulated
SNR3 = 10 * log10(mean(s2.^2) / mean(n.^2)); % reformulated

%Estimations, assuming additive noise with expected value of 0:
SNR4 = 10 * log10(mean(s2.^2) / sigma^2); % known noise variance
SNR5 = 20 * log10(norm(s2, 2) / norm(s2 - s6, 2)); % estimated noise


