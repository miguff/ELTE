%Exercise 6.
%Apply Fourier transform on the signals and interpret the results.

%This part is from exercise 3.
fs = 8000; %This will be the sampling rate (Hz)

len = 1; % length [s]
t = linspace(0, len, len*fs + 1); %This will be the sampling points. This needs the + 1, because we need 8000 values. generates n points.
t(end) = [];
fA = 440;

n = 8; %number of components
s3 = 0;
for k = 1:n
    s3 = s3 + sin(2*pi*fA*k*t) / 2^k*2;
end

%so here we are going to apply fast fourier transformation to the s3 and to
%s5

f3 = fft(s3); %Outp has the same values as input signal, but with complex numbers
figure, plot(abs(f3)) %Magnitude plot
figure, plot(angle(f3)) %Most we do not need but most of the analize application
figure, plot(abs(fftshift(f3)))  %Zero Centered

%So what is the interpretaion for this?
%frequency domain representation (cf. time domain). Simplified, peaks on the magnitude plot belongs to the sinusoidal components of the signal.


%Exercise 7.

[pks,locs] = findpeaks(abs(f3), 'NPeaks', 3, 'SortStr', 'descend'); %Here we will find the peaks in the abs(f3), we look for 3 peaks, we look for the 3 most dominant peaks
%findpeaks = findpeaks(data) returns a vector with the local maxima (peaks) of the input signal vector, data. A local peak is a data sample that is either larger than its two neighboring samples or is equal to Inf. The peaks are output in order of occurrence. Non-Inf signal endpoints are excluded. If a peak is flat, the function returns only the point with the lowest index.
%[pks,locs] = findpeaks(data) additionally returns the indices at which the peaks occur.

figure, hold on, plot(abs(f3)), plot(locs, pks, 'ro') %So here we just add red circle to the found values
hold off
%Exercis 8.
%Take the signal of tone A with its overtones (ex. 2.3.). Filter out the fundamental tone

%How to solve ot?
%Naive approach: cut out frequency components from the FFT spectrum. -->
%this is not a good approach

%Frequency sampling of the FFT - What is FTT? Fast Fourier Transform
disp(length(t))
ft = linspace(0, fs, length(t)+1); %Here it starts from 0 and creates n number of items. So if we wanted to make every number between 0 and 8000, we need the n to be 8001
ft(end) = []; %With this, we create a 8000 length array from a 8001 lenght by making the last value empty

% cutout: between 430 and 450 Hz
cut = find(ft >= 430 & ft <= 450);

%k = find(X) returns a vector containing the linear indices of each nonzero element in array X.

f8 = f3; 
f8(cut) = 0; %It 0 out the corresponding values
f8(end-cut+2) = 0; % We have to do it to be symmetrical --> cut+2 because here the first key corresponds to the last N-k+2 because of indexing and that the 0 does not have a pair. Onenote

figure, hold on, plot(abs(f8))

s8 = real(ifft(f8)); % inverse FFT
%real - real(Z) returns the real part of each
%X = ifft(Y) computes the inverse discrete Fourier transform of Y using a fast Fourier transform algorithm. X is the same size as Y.

