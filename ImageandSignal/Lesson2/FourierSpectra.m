%Exercise 10
%Let the signals be 2 Hz and 2.5 Hz sinusoids, and their sum, sampled at 30 Hz. The
%length of the measurement window varies between 1 and 5 s. Compare the Fourier
%spectra of the signals and explain the experiences.

n = 5;
Wave1 = 2;
Wave2 = 2.5;
SamplingHz = 30;

for i = 1:n %Every second
    t = linspace(0, i, SamplingHz*i+1); t(end) = []; %create the base row vector
    z1 = sin(2*pi*Wave1*t); z2 = sin(2*pi*Wave2*t); %Here we create the two sinosoid wave
    f1 = fft(z1); f2 = fft(z2); f12 = fft(z1+z2); %Here we compute the Fast Fouries tranformation on each one and on their sum
    %subplot(n, 1, i), hold on
    %figure, plot(abs(f1)), plot(abs(f2)), plot(10+abs(f12))

end

%Explanations:
%Peak distortions: Fourier boundary effect occurs if the sampling rate, the measurement window length, and the signal frequency are not proportional. Note
%that the signal energy is conserved:

%[norm(z1, 2) .^ 2, norm(z2, 2) .^ 2]
%[norm(f1, 2) .^ 2, norm(f2, 2) .^ 2] / (30 * i)

%Inseparable peaks: the resolution of the FFT depends on the measurement
%window size.

%Exercise 11
%Apply different window functions on the signals and compare the results.

t = linspace(0, 1, 30+1); t(end) = [];
z2 = sin(2*pi * 2.5 * t);
a = ones(1, 30); % rect
%b = [0:29, 30:-1:1] % tri
c = gausswin(30)'; % Gaussian
d = hamming(30)'; % Hamming
e = hann(30)'; % Hann
f = flattopwin(30)'; % flat-top
figure, hold on
plot(abs(fft(z2)))
plot(abs(fft(a .* z2)))
%plot(abs(fft(b .* z2)))
plot(abs(fft(c .* z2)))
plot(abs(fft(d .* z2)))
plot(abs(fft(e .* z2)))
plot(abs(fft(f .* z2)))


%Exercise 12
%Let the signal be a supercomposition of a 2.8 and 3.9 Hz sinusoids (rate: 30 Hz,
%length: 1 s). Make the two peaks on the Fourier spectrum separable.




Wave12_1 = 2.8;
Wave12_2 = 3.9;
SamplingHz12 = 30;
len = 1;

t = linspace(0, len, len*SamplingHz12+1); t(end) = [];
S12_1 = sin(2*pi*Wave12_1*t);
S12_2 = sin(2*pi*Wave12_2*t);

f12_1 = fft(S12_1);
f12_2 = fft(S12_2);

figure
subplot(311), hold on, plot(abs(f12_1)), plot(abs(f12_2))
subplot(312), plot(abs(fft(S12_2 + S12_1)))
w = gausswin(SamplingHz12)';
%w = gausswin(L) returns an L-point Gaussian window. Not quite sure what
%this is :P
subplot(313), plot(abs(fft(w .* (S12_2 + S12_1))))



