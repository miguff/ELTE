% Read the input image
img = imread('P9170012.jpg');

% Convert to grayscale
grayImg = rgb2gray(img);

% Apply edge detection
edges = edge(grayImg, 'Canny');

% Find contours in the binary image
[B, L] = bwboundaries(edges, 'noholes');

% Initialize variables for the license plate
licensePlateRegion = [];

% Loop through the boundaries to find rectangular objects
for k = 1:length(B)
    boundary = B{k};
    % Fit a bounding box around the contour
    x_min = min(boundary(:,2));
    x_max = max(boundary(:,2));
    y_min = min(boundary(:,1));
    y_max = max(boundary(:,1));
    
    % Calculate width and height
    width = x_max - x_min;
    height = y_max - y_min;
    
    % Check if the region looks like a license plate
    aspectRatio = width / height;
    if aspectRatio > 2 && aspectRatio < 6
        licensePlateRegion = imcrop(grayImg, [x_min, y_min, width, height]);
        break;
    end
end

% Enhance the license plate image (optional)
if ~isempty(licensePlateRegion)
    licensePlateRegion = imresize(licensePlateRegion, [300 NaN]);
    licensePlateRegion = imbinarize(licensePlateRegion);
    
    % Perform OCR
    result = ocr(licensePlateRegion, 'TextLayout', 'Block');
    disp('License Plate Number:');
    disp(result.Text);
else
    disp('License plate not detected.');
end
