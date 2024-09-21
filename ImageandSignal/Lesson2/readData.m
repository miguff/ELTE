%Exercise 9
%The sound signal in file cmajor.wav is a musical triad. Detect the frequencies of the components and decompose the signal into separated signals. Mind the overtones

%1. Step - Read the file to data
[y, fs]= audioread('cmajor.wav');
sound(y, fs)
%Here it gives us back the sampling rate (fs) and the  sampled data,

%Step 2 - Stereo -> Mono

%Here we create a Stereo from Mono, by making the of the two same sequence.
%Since we do that, we will get back one channel with the same value
y = mean(y,2);

% returns the mean along dimension dim. For example, if A is a matrix, then mean(A,2) returns a column vector containing the mean of each row.

%Step 3 - We do a Fast Fourier Transformations
f = fft(y);

%Step 4 - lower half of FFT - do not really understand why it happens as
%happens, since it creates a 33075x1 from a 66150x1
f1 = f(1:end/2); 

%Step 5 - Find the peaks and plot them
[p,l] = findpeaks(abs(f1), 'NPeaks', 3, 'SortStr', 'descend'); %Finds the first 3 peak
[l, i] = sort(l); 
p = p(i); % sorted frequency peaks
%B = sort(A) sorts the elements of A in ascending order.
%The l is the original list, that requires needs to be sorted. The output l
%is the sorted list. The i is the order of the original list
figure, hold on, plot(abs(f1)), plot(l, p, 'ro')

d = floor(min(diff(l))/4); %It filters the width

y1 = []; %It will be the output sigbal


%This parts needs a bit more studying
for i = 1:3 % I guess because we have 3 peaks
    fi = zeros(size(f1)); %Here we create a zeros vector, that has the same size as the f0
    n = floor((length(f1)-d)/l(i));

    for k = 1:n % fundamental + overtones
        fi(k*l(i)-d:k*l(i)+d) = f1(k*l(i)-d:k*l(i)+d);
    end
    fi = [fi; 0; conj(fi(end:-1:2))];
    yi = real(ifft(fi));
    y1 = [y1; yi];
end

y1 = [y1; zeros(fs/2, 1); y];
sound(y1, fs);
audiowrite('cmajor_part.wav', y1, fs);
