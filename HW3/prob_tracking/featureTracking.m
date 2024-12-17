function featureTracking
% Main function for feature tracking
folder = '.\images';
im = readImages(folder, 0:50);

tau = 0.06;                                 % Threshold for harris corner detection
[pt_x, pt_y] = getKeypoints(im{1}, tau);    % Prob 1.1: keypoint detection

ws = 7;                                     % Tracking ws x ws patches
[track_x, track_y] = ...                    % Keypoint tracking
    trackPoints(pt_x, pt_y, im, ws);
  
% Visualizing the feature tracks on the first and the last frame
figure(2), imagesc(im{1}), hold off, axis image, colormap gray
hold on, plot(track_x', track_y', 'r');
figure(3), imagesc(im{end}), hold off, axis image, colormap gray
hold on, plot(track_x', track_y', 'r');


function [track_x, track_y] = trackPoints(pt_x, pt_y, im, ws)
% Tracking initial points (pt_x, pt_y) across the image sequence
% track_x: [Number of keypoints] x [2]
% track_y: [Number of keypoints] x [2]

% Initialization
N = numel(pt_x);
nim = numel(im);
track_x = zeros(N, nim);
track_y = zeros(N, nim);
track_x(:, 1) = pt_x(:);
track_y(:, 1) = pt_y(:);

for t = 1:nim-1
    [track_x(:, t+1), track_y(:, t+1)] = ...
            getNextPoints(track_x(:, t), track_y(:, t), im{t}, im{t+1}, ws);
end


function [x2, y2] = getNextPoints(x, y, im1, im2, ws)
% Iterative Lucas-Kanade feature tracking
% x,  y : initialized keypoint position in im2
% x2, y2: tracked keypoint positions in im2
% ws: patch window size

% YOUR CODE HERE


function im = readImages(folder, nums)
im = cell(numel(nums),1);
t = 0;
for k = nums,
    t = t+1;
    im{t} = imread(fullfile(folder, ['hotel.seq' num2str(k) '.png']));
    im{t} = im2single(im{t});
end
