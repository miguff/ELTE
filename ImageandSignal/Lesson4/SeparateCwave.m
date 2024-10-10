%We have to separate the 3 waves using bandpass filter


%1. Step - Read the file to data
[y, fs]= audioread('cmajor.wav');

%Here we create a Stereo from Mono, by making the of the two same sequence.
%Since we do that, we will get back one channel with the same value. If we
%only choose the first or second channel, in other times we might lose data
y = mean(y,2);

%Step 3 - We do a Fast Fourier Transformations
f = fft(y);

%Step 4 - lower half of FFT - because Fuorier transformation is symmetric,
%so we just need the first half
% it creates a 33075x1 from a 66150x1
f1 = f(1:end/2);

%Step 5 - Find the peaks and plot them
[p,l] = findpeaks(abs(f1), 'NPeaks', 3, 'SortStr', 'descend'); %Finds the first 3 peak, it is sorted by PeakSize

[l, i] = sort(l); 
p = p(i); % sorted frequency peaks
%B = sort(A) sorts the elements of A in ascending order.
%The l is the original list, that requires needs to be sorted. The output l
%is the sorted list. The i is the order of the original list
figure, hold on, plot(abs(f1)), plot(l, p, 'ro') %If we zoom in, we can see that they are not perfect tones, it has some distortion. So we need to find more peaks potencially



%Here we compute it to frequencies
frequencys = linspace(0, fs/2, length(f1) + 1); ft(end) = []; 
figure, hold on, plot(frequencys, abs(f1)), plot(frequencys(l), p, 'ro')


freq1 = frequencys(l(1));
d = floor(min(diff(frequencys(l)))/4); %It filters the width - Measure the distance between the three peaks, and divide by 4 - it is experimental, it can be defined otherwise

N = 12;
% bandpass filter, cutoff frequencies: 500 and 600 Hz
d2 = designfilt('bandpassiir', 'FilterOrder', N, ...
'HalfPowerFrequency1', freq1-d, 'HalfPowerFrequency2', freq1+d, ...
'SampleRate', fs);
s2 = filter(d2, y);


%We should extend to the overtones



%For the butterworth and the other in the solution does not turn it to
%frequencies. Because we need the relative frequencies --> devide with half
%the sample rate