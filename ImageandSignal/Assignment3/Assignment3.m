
%% Detect the Bounding Box and get the coordinates

% Load the ONNX model
modelfile = "best.onnx";
net = importNetworkFromONNX(modelfile);

% Load and preprocess the input image
imagePath = 'test_images/P9170033.jpg'; % Replace with your image path
inputImage = imread(imagePath);
resizedImage = imresize(inputImage, [640, 640]);

% Normalize the image
inputImageNormalized = double(resizedImage) / 255; % Normalize to [0, 1]

% Predict using the model
predictions = predict(net, inputImageNormalized);

% Extract bounding boxes and confidence scores
predictions = squeeze(predictions); % Remove singleton batch dimension, resulting in [5x8400]

% Bounding boxes (x_center, y_center, width, height)
boundingBoxes = predictions(1:4, :)'; % Transpose to get [8400x4]
% Confidence scores
confidenceScores = predictions(5, :)'; % Transpose to get [8400x1]
% Best Bounding box from the Confidence Score
[~, bestIdx] = max(confidenceScores);
bestBoundingBox = boundingBoxes(bestIdx, :); % [x_center, y_center, width, height]


x_center = bestBoundingBox(1);
y_center = bestBoundingBox(2);
width = bestBoundingBox(3);
height = bestBoundingBox(4); 

x_min = x_center - width / 2;
y_min = y_center - height / 2;
x_max = x_center + width / 2;
y_max = y_center + height / 2;

% Draw the best bounding box on the original image
detectedImage = insertShape(resizedImage, 'Rectangle', [x_min, y_min, width, height], 'Color', 'red', 'LineWidth', 2);


% Display the image with the bounding box
figure;
imshow(detectedImage);
title('Detected License Plate with Best Bounding Box');



%% Get the license plate number
image = imcrop(resizedImage, [x_min, y_min, width, height]);



% Convert to grayscale
grayImage = rgb2gray(image);

% Binarize the image - Convert the grayscale image into a binary (black-and-white) image to highlight the text against the background.
% A binary image where the text should stand out as white shapes on a black background
threshold = 128;
binaryImage = grayImage > threshold;

% Perform morphological operations to isolate text
% Enhance the binary image by cleaning up noise and connecting disjointed parts of text characters.
% imdilate: Expands the white regions in the image using a structuring element (a line here).
% imerode: Shrinks the white regions, undoing the dilation slightly but removing small noise.
% strel('line', 2, 0) creates a linear structuring element of length 2 and angle 0 degrees. Enhancing horizontally-aligned text.

dilatedImage = imdilate(binaryImage, strel('line', 2, 0));
cleanedImage = imerode(dilatedImage, strel('line', 2, 0));

% Use OCR to extract license plate text
% MATLAB’s OCR function detects and recognizes text in the image.
ocrResults = ocr(cleanedImage, 'TextLayout', 'Block');

% Display results
licensePlate = ocrResults.Text;

disp('Extracted License Plate Values:');
disp(strtrim(licensePlate));

% Show cleaned image with recognized text
figure;
imshow(cleanedImage);
title(strtrim(licensePlate));
hold on;
% 
