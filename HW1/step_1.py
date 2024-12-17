from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the hokiebird image
image_path = 'hokiebird.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# Extract R, G, B values along the scanline (250th row)
row = 250
scanline_r = image_np[row, :, 0]  # Red channel
scanline_g = image_np[row, :, 1]  # Green channel
scanline_b = image_np[row, :, 2]  # Blue channel

# Plot the R, G, B values along the scanline
plt.figure(figsize=(10, 5))
plt.plot(scanline_r, color='r', label='Red')
plt.plot(scanline_g, color='g', label='Green')
plt.plot(scanline_b, color='b', label='Blue')
plt.title('R, G, B values along the 250th row')
plt.xlabel('Pixel Index')
plt.ylabel('Intensity Value')
plt.legend()
plt.savefig('01_scanline.png')
plt.show()
