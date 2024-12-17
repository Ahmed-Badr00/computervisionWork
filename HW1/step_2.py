from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')
image_np = np.array(image)

# Stack the R, G, B channels vertically
r_channel = image_np[:, :, 0]
g_channel = image_np[:, :, 1]
b_channel = image_np[:, :, 2]
stacked_rgb = np.vstack((r_channel, g_channel, b_channel))

# Save the stacked RGB image
stacked_rgb_image = Image.fromarray(stacked_rgb)
stacked_rgb_image.save('02_concat_rgb.png')
