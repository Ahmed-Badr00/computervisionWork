function [mag, theta] = gradientMagnitude(im, sigma)

im = imfilter(im, fspecial('gaussian', max(round(sigma*3)*2+1,3), 1));
[gx, gy] = gradient(im);

% compute gradient magnitude for each channel (r, g, b)
mag = sqrt(gx.^2 + gy.^2); 

% get orientation of gradient with largest magnitude
[mv, mi] = max(mag, [], 3); % max over third dimension
N = size(gy, 1)*size(gy, 2);
theta = atan2(gy((1:N) + (mi(:)'-1)*N), gx((1:N) + (mi(:)'-1)*N))+pi/2;
theta = reshape(theta, [size(gy, 1) size(gy, 2)]);

% compute overall magnitude as L2-norm of r g b
mag = sqrt(sum(mag.^2, 3));


