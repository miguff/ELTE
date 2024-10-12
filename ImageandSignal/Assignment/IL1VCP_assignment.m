%%
%Open the given file
[y, fs] = audioread('audio_13.wav');
sound(y, fs)
%We split them, we know the first half is the normal sound and the second
%half is the noisiy one
y1 = y(1:end/2);
y2 = y(end/2+1:end);


%% Use Highpass Butterworth filter 

N = 12;
Hz = 250;

d3 = designfilt('highpassiir', 'FilterOrder', N, ...
'HalfPowerFrequency', Hz, 'SampleRate', fs);

y2 = filter(d3, y2);
y = [y1; y2];
%sound(y2, fs)

%% Volume Distortion Correction
peaky1 = max(abs(y1));
peaky2 = max(abs(y2));

y2 = y2 * (peaky1 / peaky2);

y = [y1; y2];
sound(y, fs)

audiowrite('audio_13_filtered.wav', y, fs)