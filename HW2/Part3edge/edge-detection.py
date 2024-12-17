import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def gradientMagnitude(im, sigma):
    """Compute gradient magnitude and orientation."""
    if len(im.shape) == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = im

    # Smooth the image using Gaussian filter
    im_smoothed = cv2.GaussianBlur(im_gray, (0, 0), sigma)

    # Compute gradients along x and y axis
    grad_x = cv2.Sobel(im_smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im_smoothed, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude and orientation
    mag = np.sqrt(grad_x**2 + grad_y**2)
    theta = np.arctan2(grad_y, grad_x)

    return mag, theta

def orientedFilterMagnitude(im, orientations=4):
    """Compute magnitude and orientation using oriented filters."""
    if len(im.shape) == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = im

    angle_step = np.pi / orientations
    mags, thetas = [], []

    for i in range(orientations):
        angle = i * angle_step
        kernel_x = cv2.getGaborKernel((21, 21), 4.0, angle, 10.0, 0.5, 0, ktype=cv2.CV_64F)
        kernel_y = cv2.getGaborKernel((21, 21), 4.0, angle + np.pi / 2, 10.0, 0.5, 0, ktype=cv2.CV_64F)

        grad_x = cv2.filter2D(im_gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(im_gray, cv2.CV_64F, kernel_y)

        mag = np.sqrt(grad_x**2 + grad_y**2)
        theta = np.arctan2(grad_y, grad_x)

        mags.append(mag)
        thetas.append(theta)

    mags = np.stack(mags, axis=-1)
    thetas = np.stack(thetas, axis=-1)
    max_mags = np.max(mags, axis=-1)
    max_thetas = thetas[np.arange(thetas.shape[0])[:, None], np.argmax(mags, axis=-1)]

    return max_mags, max_thetas

def non_maximal_suppression(mag, theta):
    """Apply non-maximal suppression."""
    edges = cv2.Canny((mag * 255 / mag.max()).astype(np.uint8), 100, 200)
    suppressed_mag = mag * (edges > 0)
    return suppressed_mag

def edgeGradient(im, sigma=1.0):
    """Generate boundary map using simple gradient-based method."""
    mag, theta = gradientMagnitude(im, sigma)
    bmap = non_maximal_suppression(mag, theta)
    return bmap

def edgeOrientedFilters(im, orientations=4):
    """Generate boundary map using oriented filters."""
    mag, theta = orientedFilterMagnitude(im, orientations)
    bmap = non_maximal_suppression(mag, theta)
    return bmap

def save_results(im, bmap1, bmap2, filename_prefix):
    """Save input image and edge maps as files."""
    # Create output directory if it doesn't exist
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save original input image
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_input.jpg"), im)

    # Save gradient-based edge detection result
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_gradient.jpg"), 
                (bmap1 * 255 / bmap1.max()).astype(np.uint8))

    # Save oriented filter edge detection result
    cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_oriented.jpg"), 
                (bmap2 * 255 / bmap2.max()).astype(np.uint8))

def process_directory(input_dir):
    """Process all images in a directory."""
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            im = cv2.imread(image_path)

            # Ensure the image was loaded successfully
            if im is None:
                print(f"Error: Could not load image {filename}")
                continue

            print(f"Processing {filename}...")

            # Compute edge maps
            bmap1 = edgeGradient(im, sigma=1.0)
            bmap2 = edgeOrientedFilters(im, orientations=4)

            # Save results
            filename_prefix = os.path.splitext(filename)[0]
            save_results(im, bmap1, bmap2, filename_prefix)

def main():
    # Input directory containing images
    input_dir = "data/images" 

    # Process all images in the directory
    process_directory(input_dir)

if __name__ == "__main__":
    main()
