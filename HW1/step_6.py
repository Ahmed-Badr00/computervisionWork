from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')

# Convert to YCbCr color space
ycbcr_image = image.convert('YCbCr')
ycbcr_image_np = np.array(ycbcr_image)

# Extract the Y component
y_component = ycbcr_image_np[:, :, 0]

# Save the Y component image
y_image = Image.fromarray(y_component)
y_image.save('06_y_ycbcr.png')
