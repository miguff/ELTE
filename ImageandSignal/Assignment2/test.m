OriginalImage = im2double(imread('image_13.png'));
GrayScaleImage = rgb2gray(OriginalImage);

% Apply median filter on all channels
newimg = OriginalImage;
newimg(:, :, 1) = medfilt2(OriginalImage(:, :, 1), [3 3]); % Red
newimg(:, :, 2) = medfilt2(OriginalImage(:, :, 2), [3 3]); % Green
newimg(:, :, 3) = medfilt2(OriginalImage(:, :, 3), [3 3]); % Blue

% Rotate the image
I = imrotate(newimg, 1, 'bicubic', 'crop');

% Crop based on non-zero rows
rows_with_nonzero = find(any(I(:, :, 1), 2));
crop_upper = max(rows_with_nonzero);
crop_lower = min(rows_with_nonzero);
crop_diff = crop_upper - crop_lower;

rect = [1, crop_lower, size(I, 2), crop_diff]; % Crop rectangle
rotated2 = imcrop(I, rect);

% Interpolation
Vq_red = interp2(I(:, :, 1));
Vq_green = interp2(I(:, :, 2));
Vq_blue = interp2(I(:, :, 3));
Vq = cat(3, Vq_red, Vq_green, Vq_blue);

% Display results
figure;
montage({OriginalImage, I, rotated2, Vq}, 'Size', [1 4]);
