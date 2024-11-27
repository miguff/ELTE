% Read the image
image = imread('test2.jpg');

% Convert to grayscale
grayImage = rgb2gray(image);

% Binarize the image
threshold = 128;
binaryImage = grayImage > threshold;

% Perform morphological operations to isolate text
dilatedImage = imdilate(binaryImage, strel('line', 2, 0));
cleanedImage = imerode(dilatedImage, strel('line', 2, 0));

% Use OCR to extract license plate text
ocrResults = ocr(cleanedImage, 'TextLayout', 'Block');

% Display results
licensePlate = ocrResults.Text;

disp('Extracted License Plate Values:');
disp(strtrim(licensePlate));

% Show cleaned image with recognized text
figure;
imshow(cleanedImage);
title('Cleaned Image with OCR Text');
hold on;

