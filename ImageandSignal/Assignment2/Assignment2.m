
OriginalImage = im2double(imread('image_13.png'));
GrayScaleImage = rgb2gray(OriginalImage);


%Using Meadian Filter in all of its channel
newimg = OriginalImage;
newimg(:, :, 1) = medfilt2(OriginalImage(:, :, 1), [3 3]); % Red channel
newimg(:, :, 2) = medfilt2(OriginalImage(:, :, 2), [3 3]); % Green channel
newimg(:, :, 3) = medfilt2(OriginalImage(:, :, 3), [3 3]); % Blue channel

L = newimg;

H = rgb2hsv(L);
% histogram adjustment
L1 = imadjust(L, stretchlim(L)); % channel-wise
H2 = H; H2(:, :, 3) = imadjust(H2(:, :, 3));
L2 = hsv2rgb(H2); % luminance only
% histogram equalization
L3 = histeq(L); % channels together
L4 = L; % channel-wise
for i = 1:3
    L4(:, :, i) = histeq(L4(:, :, i));
end
H5 = H; H5(:, :, 3) = histeq(H5(:, :, 3));
L5 = hsv2rgb(H5); % luminance only
figure, montage({L, L1, L2, L3, L4, L5})

