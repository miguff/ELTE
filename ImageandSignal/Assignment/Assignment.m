%Input datas

%Requirement: solve the following basic signal processing task in Matlab. 
%You will be provided an audio file, where the second half of the recording is distorted. 
% The task is to enhance the quality of the audio, using the learned techniques. 
% See the table below for your personal audio file. 
% Note that each file is distorted using exactly two techniques (a structural distortion and noise).

%%
%Open the given file
[y, fs] = audioread('audio_13.wav');

%We split them, we know the first half is the normal sound and the second
%half is the noisiy one
y1 = y(1:end/2);
y2 = y(end/2+1:end);
%sound(y, fs)

%% Frequency domain analysis I.
f1 = fft(y1);
f2 = fft(y2);
f1 = f1(1:end/2);
f2 = f2(1:end/2);
ft = linspace(0, fs/2, length(f1)+1); ft(end) = [];

mask = ft < 5000;

figure

subplot(2, 1, 1)
plot(ft(mask), abs(f1(mask)))
hold on
title('Frequency Domain - Intact Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
subplot(2, 1, 2) 
plot(ft(mask), abs(f2(mask)))
title('Frequency Domain - Distorted Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
sgtitle('Fequency Doman Analysis I.', 'FontSize', 18, 'FontWeight', 'bold');
hold off
linkaxes

%Conclusion:
% We need a Highpass Butterworth filter at around 250 Hz



%% Use Highpass Butterworth filter 

N = 12;
Hz = 250;

d3 = designfilt('highpassiir', 'FilterOrder', N, ...
'HalfPowerFrequency', Hz, 'SampleRate', fs);

y2 = filter(d3, y2);
y = [y1; y2];
%sound(y2, fs)

%% Frequenc domain analysis II.
f1 = fft(y1);
f2 = fft(y2);
f1 = f1(1:end/2);
f2 = f2(1:end/2);
ft = linspace(0, fs/2, length(f1)+1); ft(end) = [];

mask = ft < 5000;

figure

subplot(2, 1, 1)
plot(ft(mask), abs(f1(mask)))
hold on
title('Frequency Domain - Intact Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
subplot(2, 1, 2) 
plot(ft(mask), abs(f2(mask)))
title('Frequency Domain - Distorted Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
hold off
sgtitle('Fequency Doman Analysis II. - After filtering', 'FontSize', 18, 'FontWeight', 'bold');
linkaxes


%% Time domain analysis I.
t = (0:length(y)-1) / fs;
t1 = t(1:end/2);
t2 = t(end/2+1:end);

figure
ax1 = subplot(3, 1, 1);
plot(t1, y1, 'b', t2, y2, 'r')
title('Time Domain Analysis', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');

ax2 = subplot(3, 1, 2);
plot(t1, y1)
title('Frequency Domain - Intact Half', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');

ax3 = subplot(3, 1, 3);
plot(t2, y2)
title('Frequency Domain - Distorted Half', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');
sgtitle('Time Domain Analysis I.', 'FontSize', 18, 'FontWeight', 'bold');
linkaxes([ax1 ax2 ax3], 'y')
% Conclusion
%   no external values => no salt-pepper noise
%   The amplitudes are not similar => possible volume distortion?

%% Volume Distortion Correction
peaky1 = max(abs(y1));
peaky2 = max(abs(y2));

y2 = y2 * (peaky1 / peaky2);

y = [y1; y2];

%% Time domain analysis II.
t = (0:length(y)-1) / fs;
t1 = t(1:end/2);
t2 = t(end/2+1:end);

figure
ax1 = subplot(3, 1, 1);
plot(t1, y1, 'b', t2, y2, 'r')
title('Time Domain Analysis', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');

ax2 = subplot(3, 1, 2);
plot(t1, y1)
title('Frequency Domain - Intact Half', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');

ax3 = subplot(3, 1, 3);
plot(t2, y2)
title('Frequency Domain - Distorted Half', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('Time (s)');
ylabel('Amplitude');
sgtitle('Time Domain Analysis II. - After Volume correction', 'FontSize', 18, 'FontWeight', 'bold');
linkaxes([ax1 ax2 ax3], 'y')



%% Frequenc domain analysis III.
f1 = fft(y1);
f2 = fft(y2);
f1 = f1(1:end/2);
f2 = f2(1:end/2);
ft = linspace(0, fs/2, length(f1)+1); ft(end) = [];

mask = ft < 5000;

figure

subplot(2, 1, 1)
plot(ft(mask), abs(f1(mask)))
hold on
title('Frequency Domain - Intact Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
subplot(2, 1, 2) 
plot(ft(mask), abs(f2(mask)))
title('Frequency Domain - Distorted Half');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
hold off
sgtitle('Fequency Doman Analysis III. - Final Version', 'FontSize', 18, 'FontWeight', 'bold');
linkaxes

audiowrite('audio_13_filtered.wav', y, fs)
