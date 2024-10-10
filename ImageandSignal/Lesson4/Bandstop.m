%BandStop we want to remove something
%We cut out a part of the signal - Completly remove a frequency range
%in 3.8 in the designfilt it should be bandstop

%Original Signaél
fs = 8000; % sampling rate
t = linspace(0, 1, fs + 1); t(end) = []; % sampling points
sigma = 0.25; % noise intensity
s1 = sin(2*pi * 440 * t); 


%Powerline line interference 
f = 50; % frequency
A = [0.25, 0.1, 0.01]; % fundamental and overtone amplitudes
s2 = s1;
for k = 1:length(A)
    s2 = s2 + A(k) * sin(2*pi * f * k * t);
end

%Remove the lowest component

N = 12; % filter order
d = designfilt('bandstopiir', 'FilterOrder', N, ...
'HalfPowerFrequency1', 40, 'HalfPowerFrequency2', 60, ...
'SampleRate', fs);
s3 = filter(d, s);

%sound(s1, fs)
%sound(s2, fs)
%sound(s3, fs)

figure, plot(abs(fft(s1)))
figure, plot(abs(fft(s2)))
figure, plot(abs(fft(s3)))