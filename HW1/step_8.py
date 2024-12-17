from PIL import Image
import numpy as np
from skimage import color 

# Load the image
image = Image.open('hokiebird.jpg')

# Convert to grayscale and YCbCr
grayscale_image = image.convert('L')
ycbcr_image = image.convert('YCbCr')
ycbcr_image_np = np.array(ycbcr_image)
y_component = ycbcr_image_np[:, :, 0]

# Convert to CIE XYZ color space
image_np = np.array(image)
cie_xyz_image_np = color.rgb2xyz(image_np / 255.0)  # Normalize before conversion
y_xyz_component = (cie_xyz_image_np[:, :, 1] * 255).astype(np.uint8)

# Stack horizontally
concat_image = np.hstack((np.array(grayscale_image), y_component, y_xyz_component))

# Save the horizontally stacked image
concat_image_save = Image.fromarray(concat_image)
concat_image_save.save('08_concat_grey.png')

