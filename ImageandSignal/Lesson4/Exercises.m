fs = 8000; % sampling rate
t = linspace(0, 1, fs + 1); t(end) = []; % sampling points
s = 0;
for f = [440, 550, 660] % sinusoid frequencies
    s = s + sin(2*pi * f * t);
end
figure, plot(abs(fft(s))) % Fourier magnitude

N = 50; % filter order
% lowpass filter, cutoff frequency: 500 Hz
d1 = designfilt('lowpassfir', 'FilterOrder', N, ...
'CutoffFrequency', 500, 'SampleRate', fs);
s1 = filter(d1, s);
figure, plot(abs(fft(s1)))
% bandpass filter, cutoff frequencies: 500 and 600 Hz
d2 = designfilt('bandpassfir', 'FilterOrder', N, ...
'CutoffFrequency1', 500, 'CutoffFrequency2', 600, ...
'SampleRate', fs);
s2 = filter(d2, s);
figure, plot(abs(fft(s2)))
% highpass filter, cutoff frequency: 600 Hz
d3 = designfilt('highpassfir', 'FilterOrder', N, ...
'CutoffFrequency', 600, 'SampleRate', fs);
s3 = filter(d3, s);
figure, plot(abs(fft(s3)))