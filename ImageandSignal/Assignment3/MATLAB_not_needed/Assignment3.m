img = imread('P9170014.jpg'); % Step 1


x_min = 300;
y_min = 370;

x_max = 400;
y_max = 100;

img = imcrop(img, [x_min, y_min, x_max, y_max]);


imshow(img)
grayImg = rgb2gray(img); % Convert to grayscale
filteredImg = medfilt2(grayImg, [3, 3]); % Median filtering
imshow(filteredImg);

edges = edge(filteredImg, 'Canny'); % Edge detection
imshow(edges);

se = strel('rectangle', [5, 5]); % Structuring element
dilatedEdges = imdilate(edges, se); % Dilate edges
imshow(dilatedEdges);

%Még nem tudom mire fog kelleni
% stats = regionprops(dilatedEdges, 'BoundingBox', 'Area');
% for k = 1:length(stats)
%     bbox = stats(k).BoundingBox;
%     aspectRatio = bbox(3) / bbox(4);
%     if aspectRatio > 2 && aspectRatio < 6 % Aspect ratio of license plates
%         licensePlateRegion = imcrop(grayImg, bbox);
%         figure, imshow(licensePlateRegion);
%          % Assuming only one plate
%     end
% end