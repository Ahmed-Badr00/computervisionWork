from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')
image_np = np.array(image)

# Create the mask
mask = (image_np > 127).astype(np.uint8) * 255

# Save the mask image
mask_image = Image.fromarray(mask)
mask_image.save('11_mask.png')
