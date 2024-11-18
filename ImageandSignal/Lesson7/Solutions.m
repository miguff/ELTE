%% 7.1 Generate a black-and-white checkerboard image. Compute and display the twodimensional Fourier transform. Apply log transform, and use color maps for better visibility. Interpret the results.

B = checkerboard(64); %512-512, black-white-gray checkerboard
B = double(B > 0); %Black and White
F = fftshift(fft2(B)); %2D FFT, centered
figure, imshow(abs(F), []) % magnitude, normalized
F1 = log(1 + abs(F)); % log transform
figure, imshow(F1, []), colormap jet, colorbar, truesize
%Interpretation: the periodic tile pattern appears as a grid of spikes on the 2D Fourier spectrum. Note that the number of tiles can be estimated from the grid size.

%% 7.2 Compute the Fourier transform of the grayscale Lenna (lena.png). Reconstruct the image using the FFT magnitude, and using the phase only. Explain the experiences.

L = rgb2gray(im2double(imread('lena.png'))); % grayscale Lenna
F = fft2(L); % 2D FFT
L1 = real(ifft2(abs(F))); % magnitude only
L2 = real(ifft2(exp(1i*angle(F)))); % phase only
L2 = rescale(L2); % normalization
figure, montage({L, L1, L2}, 'Size', [1 3])
%Explanation: the phase of the Fourier spectrum stores the detail information of the image

%% 7.3 . The image A2.png contains a repeating pattern. Detect and remove the pattern with the Fourier transform.

A = im2double(imread('A2.png')); %Read the image
figure, imshow(A)
A = A - mean(A(:)); % remove DC component
figure, imshow(A)

%The DC component of an image refers to the average intensity or brightness of the image and represents the zero-frequency component of its frequency spectrum.

%Fourier Transform Context:
%In the context of a 2D Fourier Transform, an image is represented as a combination of different spatial frequencies. The DC component is located at the center of the Fourier spectrum (at zero frequency).
%This component corresponds to the mean of all pixel intensities in the spatial domain. It does not carry any information about spatial variations or details in the image but represents its overall brightness.

%In Practice:
%Removing the DC component from an image (by subtracting the mean intensity) results in a centered image where the average intensity is zero. This is useful in certain image processing tasks, such as enhancing contrast or focusing on high-frequency details.
F = fft2(A); % 2D FFT
F1 = abs(F); % FFT magnitude
F2 = fftshift(log(1 + F1)); % log transform
figure, surf(F2), shading interp % surface display
F(F1 == max(F1(:))) = 0; % peak removal
A1 = real(ifft2(F)); % reconstruction
figure, imshow(A1, []) % normalized display

%%
