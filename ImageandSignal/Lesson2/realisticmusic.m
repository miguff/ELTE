%Make the sound signals more realistic: add a linear fade-in and fade-out effect.
fs = 8000; %This will be the sampling rate (Hz)

len = 1; % length [s]
t = linspace(0, len, len*fs + 1); %This will be the sampling points. This needs the + 1, because we need 8000 values. generates n points.
t(end) = [];
fA = 440;
s1 = sin(2*pi*t*fA);


len5 = floor(0.2*fs); %200 ms will be the duration. floor = rounds each element of X to the nearest integer less than or equal to that element.

w = ones(size(t)); %Here we create the amplitude mask. It creates a t sized vector that contains only ones. Here this is a 1 row t column vector
w(1:len5) = linspace(0, 1, len5); %fade in - it means that we have a 1 second lenght sound, which is has an 8000 sampling points.
% Here from that 8000 for the first 200 ms, we will fade, which means we
% will create a 0 to 1 vector with 200ms long steps, here it is len5, so
% for the first 1600 sampling points will be equally distributed point form
% 0 to 1 and from on that the last 6400 points will be one

%Check their length to know many we want to fade
length(w(end-len5+1:end))
length(linspace(1, 0, len5))


%Here we did exactly the opposit, for the last 1600 sampling points, it
%will decrease to 0
w(end-len5+1:end) = linspace(1, 0, len5);
figure, plot(w)
s5 = w .* s1; %Here we generate a sinus wave from the original sinus wave that fades in and fades out, it is now has an amplitude

% * is a matrix multiplication
% .* is an element-wise multiplication

figure, plot(s5)
sound(s5, fs)










