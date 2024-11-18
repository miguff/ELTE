%% 8.1  Simulate and filter electric noise. Add white Gaussian noise to the grayscale Lenna (lena.png), and filter it with moving average and Gaussian filters.
%Generation: random matrix with distribution N (0, σ2):

%Make the image noisy
sigma = 0.1;
L = rgb2gray(im2double(imread('lena.png')));
N = sigma * randn(size(L)); % white noise
L1 = L + N; % noisy image
figure, montage({L, L1})


%% Filter with moving average filter
K = ones(5) / 25; % 5x5 kernel - The greater the divider, the darker the image
L2 = imfilter(L1, K); % convolutional filter
%figure, montage({L, L1, L2}, 'Size', [1 3])

%Filtering with Gaussian filter:
fsigma = 1; % filter std
K = fspecial('gaussian', 5, fsigma); % 5x5 kernel
L3 = imfilter(L1, K);
%figure, montage({L, L1, L2, L3}, 'Size', [1 4])

% One line with matlab
L4 = imgaussfilt(L1, fsigma); % Gaussian filter
figure, montage({L, L1, L2, L3, L4}, 'Size', [1 5])


%% 8.2 Handle the boundary problem.
K = fspecial('average', 25); % large kernel
L2 = imfilter(L1, K); % default: zero padding
L3 = imfilter(L1, K, 'replicate'); % replicate border
L4 = imfilter(L1, K, 'symmetric'); % symmetric extension
L5 = imfilter(L1, K, 'circular'); % periodic extension
figure, montage({L2, L3, L4, L5}, 'Size', [2 2])

%Notice the artifical border in case of zero padding and periodic
%extension. 

%% 8.3 
%Simulate and filter photon noise (shot noise). Add Poisson noise to the grayscale
%Lenna, and filter it with Gaussian filter.

L2 = im2double(imnoise(im2uint8(L), 'poisson'));
figure, montage({L, L1, L2}, 'Size', [1 3])

sigma = 0.1; % noise intensity
L1 = imnoise(L, 'localvar', (sigma * L).^2);
figure, montage({L, L1})
% Filtering with 8.1

%% 8.4  Simulate and filter detector malfunction. Add salt-pepper noise to the grayscale Lenna, and filter it with moving median filter.

d = 0.01; % noise density (1%)
L1 = imnoise(L, 'salt & pepper', d);
%figure, montage({L, L1})

%Gaussian filtering
fsigma = 1;

%Moving average filter
K = ones(5) / 25;

%Filter with moving meadian filter
L2 = medfilt2(L1, [3 3]); % 3x3 window median
L3 = imgaussfilt(L1, fsigma);
L4 = imfilter(L1, K);


figure, montage({L, L1, L2, L3, L4}, 'Size', [1 5])

%% 8.5   Estimate the signal-to-noise ratio (SNR), and compute the structural similarity (SSIM) index between the original and noisy images. SNR with known and estimated noise:

10 * log10(mean(L1(:).^2) / mean(N(:).^2)) % known noise
10 * log10(mean(L1(:).^2) / sigma^2) % known noise variance
10 * log10(mean(L1(:).^2) / mean((L1(:)-L2(:)).^2)) % estimated
ssim(L, L1)


%% 8.6 The file lena_dist.png is a distorted version of Lenna (grayscale). Apply filters and contrast enhancement methods to improve its visual interpretation.
L = im2double(imread('lena_dist.png'));
L1 = medfilt2(L, [3 3]);
L2 = imadjust(L1);
L3 = imgaussfilt(L2, 1);
L4 = imadjust(L3);
figure, montage({L, L1, L2, L3, L4}, 'Size', [1 5])



%% 8.7 The folder lena_noise contains noise measurements of the original Lenna (grayscale, white Gaussian noise). Restore Lenna from these images.

%Idea: motivated by the law of large numbers, compute the average of the images (temporal average).

list = dir('lena_noise\*.png');
L = 0;
for i = 1:length(list)
    L = L + im2double(imread(['lena_noise\', list(i).name]));
end
L = L / length(list);
figure, imshow(L)


%% 8.8 
L = im2double(imread('lena.png'));
% blurring
sigma = 3; % filter std
L1 = imgaussfilt(L, sigma); % Gaussian filter
% sharpening
M = L - L1; % unsharp mask
L2 = L + M; % sharpened
figure, montage({L, L1, L2}, 'Size', [1 3])



%% 8.9 Compare the side effects (artifacts) of Gaussian and median filtering on the MATLAB image blobs.png
B = im2double(imread('blobs.png'));
% Gaussian filter
B1 = {B};
for s = [1 2 4]
    B1 = [B1, imgaussfilt(B, s)];
end
figure, montage(B1)
% median filter
B2 = {B};
for s = [3 5 9]
    B2 = [B2, medfilt2(B, [s s])];
end
figure, montage(B2)

%% 8.11
A = im2double(imread('A4.png'));
B = adapthisteq(A);
figure, montage({A, B})
% inspect the first lines (background gradient)
a = A(1, :); b = B(1, :);
N = length(a); x = 1:N;
figure, plot(x, a, x, b)
% remove linear trend
p = robustfit(x, b);
b = b - polyval(flip(p), x);
figure, plot(x, b)
% estimate periodicity
[p, f] = periodogram(b, [], [], N);
[m, k] = findpeaks(p, f, 'NPeaks', 1, 'SortStr', 'descend');
figure, plot(f, p, k, m, 'o')
fprintf('Estimated number of tiles: %d\n', round(k))



%% 8.12 Apply gradient filters to the grayscale Lenna. Compare the behavior of the filters in the presence of noise.
K = fspecial('prewitt'); % Prewitt kernel
K = fspecial('sobel'); % Sobel kernel
L = rgb2gray(im2double(imread('lena.png')));
GY = imfilter(L, K); % vertical gradients
GX = imfilter(L, K'); % horizontal gradients
GM = sqrt(GX.^2 + GY.^2); % gradient magnitude
GA = atan2(GY, GX); % gradient direction
figure, montage({rescale(GX), rescale(GY), ...
rescale(GM), rescale(GA)})



%% 8.13 Apply Laplacian filter to the grayscale Lenna. Sharpen the image with the Laplacian.
K = fspecial('laplacian', 0); % 4-connected
K = fspecial('laplacian', 0.5) * 3; % 8-connected
L = rgb2gray(im2double(imread('lena.png')));
D = imfilter(L, K);
figure, imshow(D, []), colorbar, truesize
L1 = L - D; % image - Laplacian
figure, montage({L, L1})



%% 8.14 Apply Laplacian-of-Gaussian (LoG) filter to the original and noisy grayscale Lenna. Compare different noise and filter variances.


sigma = [0 0.1 0.25]; % noise std
fsigma = [1 2 4]; % filter std
L = rgb2gray(im2double(imread('lena.png')));
LL = {}; % list of images
for s1 = sigma
    L1 = imnoise(L, 'gaussian', 0, s1.^2); % noisy image
    LL = [LL, L1];
    for s2 = fsigma
        K = fspecial('log', 6*s2+1, s2); % LoG kernel
        L2 = imfilter(L1, K); % LoG filter
        LL = [LL, rescale(L2)];
    end
end
figure, montage(LL, 'Size', [3 4])

%% 8.15

K1 = fspecial('gaussian', 3, 0.5); % Gaussian kernel
K2 = fspecial('laplacian', 0); % Laplacian kernel
L = im2double(imread('lena.png'));
[M,N,~] = size(L);
P1 = zeros(2*M, 2*N); % Gaussian pyramid
P2 = zeros(2*M, 2*N); % Laplacian pyramid
L1 = L;
while min(M, N) >= 1
    [M,N,~] = size(L1);
    L2 = imfilter(L1, K2); % Laplacian
    Z = zeros(M, N);
    P1(1:2*M, 1:2*N) = [Z, L1(:, :, 1); L1(:, :, 2), L1(:, :, 3)];
    P2(1:2*M, 1:2*N) = [Z, L2(:, :, 1); L2(:, :, 2), L2(:, :, 3)];
    L1 = imfilter(L1, K1); % Gaussian
    L1 = L1(2:2:end, 2:2:end, :); % downsample
end
figure, montage({P1, P2})
