
OriginalImage = im2double(imread('image_13.png'));
GrayScaleImage = rgb2gray(OriginalImage);


%Here we apply median filter in all the channels
newimg = OriginalImage;
newimg(:, :, 1) = medfilt2(OriginalImage(:, :, 1), [3 3]); % Red
newimg(:, :, 2) = medfilt2(OriginalImage(:, :, 2), [3 3]); % Green
newimg(:, :, 3) = medfilt2(OriginalImage(:, :, 3), [3 3]); % Blue

%Roatate the image
I = imrotate(newimg,1,'bicubic','crop');


%Crop the image, so that the black background can not be seeable
crop_upper = max(find(diag(I(:, :, 1)) ~= 0));
crop_lower = min(find(diag(I(:, :, 1)) ~= 0));
crop_diff = crop_upper-crop_lower;

cropped = imcrop(I, [crop_lower, crop_lower, crop_diff, crop_diff]);
 
%Show the original and the rotated
figure, montage({OriginalImage, I}, 'Size',[1 2])

%show the cropped
figure, imshow(cropped)


