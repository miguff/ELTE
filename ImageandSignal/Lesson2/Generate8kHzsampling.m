%. Generate and play a digital sound signal corresponding to the musical tone A (440 Hz). Let the sampling rate be 8 kHz (phone quality), and the length of the signal be 1 second.

%This was the first task, no clue how to do it :D

%General form of sine waves (sinusoids)

% y(t) = A sin(2πf t + ϕ) (t ∈ R)

% where A is the amplitude, f is the (ordinary) frequency, and ϕ is the phase



%Exercise 1
fs = 8000; %This will be the sampling rate (Hz)
len = 1; %We need to create a 1 sec long note
t = linspace(0, len, len*fs+1); %This will be the sampling points. This needs the + 1, because we need 8000 values. generates n points.
t(end) = [];
fA = 440; %This is the required frequenry for the tone A

s1 = sin(2*pi*fA*t); % returns the sine of the elements of X

%Little bit disturbing when running multiple times :D
%sound(s1, fs) %sound(y,Fs) sends audio signal y to the speaker at sample rate Fs.


%Exercise 2.
%Generate a signal corresponding to the A major triad: the superposition of tones A (440 Hz), C# (550 Hz), and E (660 Hz).

fCSharp = 550;
fE = 660;

%It looks if you want to create the superposition of tones, you need to add
%the rewuired toe frequencies together. This is still an 8 kHz sampling
%rate, since the 't' contains th fs
s2 = s1 + sin(2*pi*fCSharp*t) + sin(2*pi*fE*t);

%Little bit disturbing when running multiple times :D
%sound(s2, fs)

% Exercise 3.
%Combine the fundamental tone A with some of its overtones. Make the amplitudes of the overtones exponentially decreasing


n = 8; %number of components
s3 = 0;
for k = 1:n
    disp(2^k*2)
    s3 = s3 + sin(2*pi*fA*t) / 2^k*2; %pi*A is a overtone, We divide 2^k-1 to decrease
end

%Little bit disturbing when running multiple times :D
%sound(s3, fs)

%Exercise 4.
%Generate multiple signals of tone A with different phase. Play them separately and combined. Explain the experiences.

%Generate multiple signals of tone A with different phase
s4_1 = sin(2*pi*fA*t+0.5);
s4_2 = sin(2*pi*fA*t+0.75);
s4_3 = sin(2*pi*fA*t+1);
s4_4 = sin(2*pi*fA*t+1.25);

s_combined = s1 + s4_1 + s4_2 + s4_3 +s4_4;
sound(s_combined, fs)

%Explanation: interference occurs. Special cases: constructive (ϕ = 2kπ) and destructive (ϕ = (2k + 1)π). In general, the result is a sinusoid with the same frequency:
% sin(2*pi*f*t) + sin(2*pi*f*t+phi) = 2*cos(phi/2)*sin(2*pi*f*t +(phi/2))





