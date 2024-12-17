import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from typing import Tuple, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridImageGenerator:
    def __init__(self, output_dir: str = "hybrid_results"):
        """Initialize the Hybrid Image Generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def low_pass_filter(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply a low-pass filter using Gaussian blur."""
        return gaussian_filter(image, sigma=sigma)
    
    def high_pass_filter(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply a high-pass filter by subtracting the low-pass image."""
        return image - self.low_pass_filter(image, sigma)
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Resize second image to match the size of the first."""
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    def determine_sigma(self, content_type: str) -> Tuple[float, float]:
        """Determine sigma values based on content type."""
        if content_type == 'faces':
            return 6, 3
        elif content_type == 'objects':
            return 8, 4
        elif content_type == 'textures':
            return 4, 2
        else:
            return 5, 3
    
    def create_hybrid_image(self, 
                          img1: np.ndarray, 
                          img2: np.ndarray, 
                          sigma1: float, 
                          sigma2: float) -> np.ndarray:
        """Create a hybrid image."""
        low_pass = self.low_pass_filter(img1, sigma1)
        high_pass = self.high_pass_filter(img2, sigma2)
        return np.clip(low_pass + high_pass, 0, 255)
    
    def create_hybrid_scale_visualization(self, 
                                       hybrid_image: np.ndarray, 
                                       num_scales: int = 5) -> np.ndarray:
        """Create a visualization of the hybrid image at different scales."""
        scales = []
        cur_img = hybrid_image.copy()
        
        for scale in range(num_scales):
            scales.append(cur_img)
            cur_img = cv2.resize(cur_img, (cur_img.shape[1] // 2, cur_img.shape[0] // 2))
        
        max_h = scales[0].shape[0]
        total_width = sum(img.shape[1] for img in scales)
        
        # Create the correct shape for grayscale or color
        if len(hybrid_image.shape) == 3:
            vis_image = np.zeros((max_h, total_width, hybrid_image.shape[2]), dtype=hybrid_image.dtype)
        else:
            vis_image = np.zeros((max_h, total_width), dtype=hybrid_image.dtype)
        
        x_offset = 0
        for img in scales:
            y_offset = (max_h - img.shape[0]) // 2
            if len(img.shape) == 3:
                vis_image[y_offset:y_offset + img.shape[0], 
                         x_offset:x_offset + img.shape[1], :] = img
            else:
                vis_image[y_offset:y_offset + img.shape[0], 
                         x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
            
        return vis_image
    
    def save_fft_visualization(self, 
                             image: np.ndarray, 
                             filename: str, 
                             title: str):
        """Save FFT visualization."""
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.colorbar(label='Log Magnitude')
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def process_image_pair(self,
                          img1_path: str,
                          img2_path: str,
                          pair_name: str,
                          content_type: str) -> None:
        """Process a pair of images to create hybrid images and visualizations."""
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            logger.error(f"Failed to load images: {img1_path}, {img2_path}")
            return
        
        # Align images
        img2 = self.align_images(img1, img2)
        
        # Determine sigma values
        sigma1, sigma2 = self.determine_sigma(content_type)
        
        # Create filtered images
        low_pass = self.low_pass_filter(img1, sigma1)
        high_pass = self.high_pass_filter(img2, sigma2)
        
        # Create hybrid image
        hybrid_image = self.create_hybrid_image(img1, img2, sigma1, sigma2)
        
        # Create scale visualization
        scale_vis = self.create_hybrid_scale_visualization(hybrid_image)
        
        # Display and save results
        images = [img1, low_pass, img2, high_pass, hybrid_image]
        titles = [
            f"{pair_name} - Image 1",
            f"{pair_name} - Low-Pass",
            f"{pair_name} - Image 2",
            f"{pair_name} - High-Pass",
            f"{pair_name} - Hybrid"
        ]
        filenames = [
            f"{pair_name}_1.png",
            f"{pair_name}_lowpass.png",
            f"{pair_name}_2.png",
            f"{pair_name}_highpass.png",
            f"{pair_name}_hybrid.png"
        ]
        
        # Save individual images
        plt.figure(figsize=(20, 4))
        for i, (img, title, filename) in enumerate(zip(images, titles, filenames)):
            plt.subplot(1, 5, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            cv2.imwrite(os.path.join(self.output_dir, filename), img)
        
        plt.savefig(os.path.join(self.output_dir, f"{pair_name}_results.png"))
        plt.close()
        
        # Save scale visualization
        plt.figure(figsize=(20, 4))
        plt.imshow(scale_vis, cmap='gray')
        plt.title(f"{pair_name} - Scale Visualization")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"{pair_name}_scales.png"))
        plt.close()
        
        # Save FFT visualizations
        self.save_fft_visualization(img1, f"{pair_name}_fft1.png", f"FFT of {pair_name} - Image 1")
        self.save_fft_visualization(low_pass, f"{pair_name}_fft_lowpass.png", f"FFT of {pair_name} - Low-Pass")
        self.save_fft_visualization(img2, f"{pair_name}_fft2.png", f"FFT of {pair_name} - Image 2")
        self.save_fft_visualization(high_pass, f"{pair_name}_fft_highpass.png", f"FFT of {pair_name} - High-Pass")
        self.save_fft_visualization(hybrid_image, f"{pair_name}_fft_hybrid.png", f"FFT of {pair_name} - Hybrid")

# Define image pairs
image_pairs = [
    ('data/marilyn.bmp', 'data/einstein.bmp', 'marilyn_einstein', 'faces'),
    ('data/fish.bmp', 'data/submarine.bmp', 'fish_submarine', 'objects'),
    ('data/bicycle.bmp', 'data/motorcycle.bmp', 'bicycle_motorcycle', 'objects'),
    ('data/bird.bmp', 'data/plane.bmp', 'bird_plane', 'objects'),
    ('data/makeup_before.jpg', 'data/makeup_after.jpg', 'makeup_before_after', 'faces'),
    ('data/Afghan_girl_before.jpg', 'data/Afghan_girl_after.jpg', 'afghan_girl', 'faces'),
    ('data/cat.bmp', 'data/dog.bmp', 'cat_dog', 'objects')
]

if __name__ == "__main__":
    generator = HybridImageGenerator()
    
    # Process all image pairs
    for img1_path, img2_path, pair_name, content_type in image_pairs:
        logger.info(f"Processing {pair_name}")
        generator.process_image_pair(img1_path, img2_path, pair_name, content_type)
    
    logger.info(f"Results saved to {generator.output_dir}")
