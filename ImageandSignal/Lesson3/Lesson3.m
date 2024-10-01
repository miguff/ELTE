%Here are the exercises for the 3. week exercises


%Fuorier Transform is a continoius sequence, but if we take the wave it
%will probably wont be contonouos, there will be a jump between the two
%parts where it was cut down - Discontinuoity


%Solutions:

%1. Adjust the windows: Only works if there is only one sine wave
%2. Easiest is to fade in and fade out, so we put everything down in 0
%zero, so the start and the end will be continouos


%Here are some of the window functions: 
%Square
%Triangle
%Gaussian



%Filter - Remove Noise, Analize the signal - These are the two most known


%Exercise 1.
%Noise type - Instrumental Noise - Record your signal - Electric device has
%noise - It can be modeled with Gaussian noise
%Simulate and filter electric instrumental noise. Generate a sine wave (e.g. a musicaltone) and add white Gaussian noise. Filter it with moving average and Gaussian filters.

Heartz = 440;
fs = 8000; % sampling rate
t = linspace(0, 1, fs + 1); t(end) = []; % sampling points
s1 = sin(2*pi * Heartz * t);


%Generate a random gaussian noise
sigma = 0.25;
n = sigma * randn(size(s1));
s2 = s1 + n;
%sound([s1, s2, n], fs) %Nois4 + Signal together

%The blue is the correct sin wave, and the red is the noise, the size is
%similar but more random


N = 5; %This is going to be our window size - It means that it will keep, or take the average of the value inside this window
s3 = zeros(size(s2)); %This is going to be a filtered signal, it means that we will replace the 0 values with the average value
length(s2)
for i = 1:length(s2)-N+1 %This is walk over every data, until the we can create the last window
    s3(i) = mean(s2(i:i+N-1)); %Here we take the window (5) average - The -1 is that i =1 and i+N = 6, this is 6 value that we have to count, but we need just 5
end


%But this is straight Convolutional, so we can use the MATLAB conv function
w = ones(1, N) /N; %divide it by the N
s4 = conv(s2, w, "same");


w = [1, 2, 1] / 4; % integer kernel of size 3
w = [1, 4, 6, 4, 1] / 16; % integer kernel of size 3
w = gausswin(5); w = w / sum(w); % built-in function
N = length(w);
s3 = zeros(size(s2));

%I have to check why the Gaussian not working
% for i = 1:length(s2)-N+1
%     s3(i) = sum(w .* s2(i:i+N-1)); % weighted sum
% end
%sound([s1, s2, s3], fs)

s4 = conv(s2, w, 'same');
s5 = filter(w, 1, s2);


%Exercise 2. Here we are going to compare them
figure, plot(t(1:100), s1(1:100), t(1:100), s3(1:100)) % left shift
figure, plot(t(1:100), s1(1:100), t(1:100), s4(1:100)) % centered
figure, plot(t(1:100), s1(1:100), t(1:100), s5(1:100)) % right


%Padding size should be half of Kernel size  I guess max








