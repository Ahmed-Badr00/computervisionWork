from PIL import Image
import numpy as np

# Load the grayscale image
image = Image.open('hokiebird.jpg').convert('L')
image_np = np.array(image)

# Initialize the non-max suppression image
non_max_image = np.zeros_like(image_np)

# Perform non-maximum suppression over a 5x5 window
for i in range(2, image_np.shape[0] - 2):
    for j in range(2, image_np.shape[1] - 2):
        window = image_np[i-2:i+3, j-2:j+3]
        max_value = np.max(window)
        non_max_image[i-2:i+3, j-2:j+3] = np.where(window == max_value, 255, non_max_image[i-2:i+3, j-2:j+3])

# Save the result
non_max_image_save = Image.fromarray(non_max_image)
non_max_image_save.save('13_nonmax.png')
