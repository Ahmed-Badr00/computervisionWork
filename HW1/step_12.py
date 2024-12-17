from PIL import Image
import numpy as np

# Load the image and mask
image = Image.open('hokiebird.jpg')
image_np = np.array(image)
mask = (image_np > 127).astype(np.uint8) * 255

# Compute the mean R, G, B values
r_channel = image_np[:, :, 0]
g_channel = image_np[:, :, 1]
b_channel = image_np[:, :, 2]

mean_r = np.mean(r_channel[mask[:, :, 0] == 255])
mean_g = np.mean(g_channel[mask[:, :, 1] == 255])
mean_b = np.mean(b_channel[mask[:, :, 2] == 255])

# Write the mean R, G, B values to a text file
with open('12_mean_rgb.txt', 'w') as f:
    f.write(f'Mean R: {mean_r}\n')
    f.write(f'Mean G: {mean_g}\n')
    f.write(f'Mean B: {mean_b}\n')

print(f'Mean R: {mean_r}, Mean G: {mean_g}, Mean B: {mean_b}')
