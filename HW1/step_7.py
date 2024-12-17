from PIL import Image
import numpy as np
from skimage import color

# Load the image
image = Image.open('hokiebird.jpg')
image_np = np.array(image)

# Convert to CIE XYZ color space
cie_xyz_image_np = color.rgb2xyz(image_np / 255.0)
y_xyz_component = (cie_xyz_image_np[:, :, 1] * 255).astype(np.uint8)

# Save the Y component image
y_xyz_image = Image.fromarray(y_xyz_component)
y_xyz_image.save('07_y_xyz.png')
