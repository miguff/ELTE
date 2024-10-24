%Exercise 9
%The sound signal in file cmajor.wav is a musical triad. Detect the frequencies of the components and decompose the signal into separated signals. Mind the overtones
%Here we have the Fundamental tones, and the overtunes. And here we have
%Three different Tones, that we want to separeta


%1. Step - Read the file to data
[y, fs]= audioread('cmajor.wav');
sound(y, fs)
%Here it gives us back the sampling rate (fs) and the  sampled data,

%Step 2 - Stereo -> Mono

%Here we create a Stereo from Mono, by making the of the two same sequence.
%Since we do that, we will get back one channel with the same value. If we
%only choose the first or second channel, in other times we might lose data
y = mean(y,2);

% returns the mean along dimension dim. For example, if A is a matrix, then mean(A,2) returns a column vector containing the mean of each row.

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

d = floor(min(diff(l))/4); %It filters the width - Measure the distance between the three peaks, and divide by 4 - it is experimental, it can be defined otherwise

y1 = []; %It will be the output sigbal


%This parts needs a bit more studying
for i = 1:3 % I guess because we have 3 peaks
    fi = zeros(size(f1)); %Here we create a zeros vector, that has the same size as the f0 - It will be an empty furier transforms
    n = floor((length(f1)-d)/l(i)); 

    for k = 1:n % fundamental + overtones
        fi(k*l(i)-d:k*l(i)+d) = f1(k*l(i)-d:k*l(i)+d); %FRom the original Fourier transforms to this new transform
    end
    fi = [fi; 0; conj(fi(end:-1:2))];
    yi = real(ifft(fi));
    y1 = [y1; yi];
end

y1 = [y1; zeros(fs/2, 1); y]; %Here there will be a short pause between the the tones. The y1 is the separated and to y is the original
sound(y1, fs);
audiowrite('cmajor_part.wav', y1, fs);









