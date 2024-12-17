import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from typing import List, Tuple

class ImagePyramidAnalyzer:
    def __init__(self, output_dir: str = "pyramid_results"):
        """Initialize the pyramid analyzer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_gaussian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """Create a Gaussian pyramid of specified levels.
        
        Args:
            image: Input grayscale image
            levels: Number of pyramid levels
            
        Returns:
            List of images forming the Gaussian pyramid
        """
        pyramid = [image.copy()]
        current_image = image.copy()
        
        for _ in range(levels - 1):
            # Apply Gaussian blur
            blurred = gaussian_filter(current_image, sigma=1)
            # Downsample by factor of 2
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)
            current_image = downsampled
            
        return pyramid
    
    def create_laplacian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """Create a Laplacian pyramid of specified levels.
        
        Args:
            image: Input grayscale image
            levels: Number of pyramid levels
            
        Returns:
            List of images forming the Laplacian pyramid
        """
        gaussian_pyramid = self.create_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            # Get current and next Gaussian level
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            
            # Upsample next level
            upsampled = np.zeros((current.shape[0], current.shape[1]), dtype=np.float32)
            upsampled[::2, ::2] = next_level
            upsampled = gaussian_filter(upsampled, sigma=1)
            
            # Compute Laplacian as difference
            laplacian = current - upsampled
            laplacian_pyramid.append(laplacian)
            
        # Add the last Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def compute_fft_amplitude(self, image: np.ndarray) -> np.ndarray:
        """Compute and return the FFT amplitude spectrum."""
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        return np.log(np.abs(fft_shift) + 1)
    
    def display_pyramid(self, 
                       pyramid: List[np.ndarray], 
                       title: str,
                       filename: str,
                       is_laplacian: bool = False):
        """Display and save a pyramid visualization.
        
        Args:
            pyramid: List of pyramid levels
            title: Title for the plot
            filename: Output filename
            is_laplacian: Whether this is a Laplacian pyramid
        """
        n_levels = len(pyramid)
        plt.figure(figsize=(20, 4))
        
        for i, level in enumerate(pyramid):
            plt.subplot(1, n_levels, i + 1)
            
            if is_laplacian and i < n_levels - 1:
                # For Laplacian levels (except last), normalize to [-1, 1]
                level_normalized = level / np.max(np.abs(level))
                plt.imshow(level_normalized, cmap='gray')
            else:
                # For Gaussian levels and last Laplacian level
                plt.imshow(level, cmap='gray')
                
            plt.title(f'Level {i}')
            plt.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def display_fft_amplitudes(self, 
                             pyramid: List[np.ndarray], 
                             title: str,
                             filename: str):
        """Display and save FFT amplitudes of pyramid levels.
        
        Args:
            pyramid: List of pyramid levels
            title: Title for the plot
            filename: Output filename
        """
        n_levels = len(pyramid)
        plt.figure(figsize=(20, 4))
        
        for i, level in enumerate(pyramid):
            plt.subplot(1, n_levels, i + 1)
            
            # Compute FFT amplitude
            fft_amplitude = self.compute_fft_amplitude(level)
            
            # Display with appropriate scaling
            plt.imshow(fft_amplitude, cmap='viridis')
            plt.colorbar()
            plt.title(f'Level {i} FFT')
            plt.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def analyze_image(self, image_path: str, levels: int = 5):
        """Perform complete pyramid analysis on an image.
        
        Args:
            image_path: Path to input image
            levels: Number of pyramid levels
        """
        # Load and convert image to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Ensure image is float32 for processing
        image = image.astype(np.float32) / 255.0
        
        # Create pyramids
        gaussian_pyramid = self.create_gaussian_pyramid(image, levels)
        laplacian_pyramid = self.create_laplacian_pyramid(image, levels)
        
        # Display pyramids
        self.display_pyramid(
            gaussian_pyramid,
            "Gaussian Pyramid",
            "gaussian_pyramid.png"
        )
        self.display_pyramid(
            laplacian_pyramid,
            "Laplacian Pyramid",
            "laplacian_pyramid.png",
            is_laplacian=True
        )
        
        # Display FFT amplitudes
        self.display_fft_amplitudes(
            gaussian_pyramid,
            "Gaussian Pyramid FFT Amplitudes",
            "gaussian_pyramid_fft.png"
        )
        self.display_fft_amplitudes(
            laplacian_pyramid,
            "Laplacian Pyramid FFT Amplitudes",
            "laplacian_pyramid_fft.png"
        )

# Example usage
if __name__ == "__main__":
    analyzer = ImagePyramidAnalyzer()
    
    # Process an example image (replace with your image path)
    image_path = "picture.jpg"
    analyzer.analyze_image(image_path, levels=5)