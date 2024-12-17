from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')
image_np = np.array(image)

# Swap the red and green channels
swapped_image_np = image_np.copy()
swapped_image_np[:, :, 0] = image_np[:, :, 1]  # Red to Green
swapped_image_np[:, :, 1] = image_np[:, :, 0]  # Green to Red

# Save the swapped image
swapped_image = Image.fromarray(swapped_image_np)
swapped_image.save('03_swapchannel.png')
