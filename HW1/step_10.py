from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')

# Crop the image to 372x372
crop_size = 372
cropped_image = image.crop((0, 0, crop_size, crop_size))

# Rotate by 90, 180, 270 degrees
rotated_90 = cropped_image.rotate(90)
rotated_180 = cropped_image.rotate(180)
rotated_270 = cropped_image.rotate(270)

# Stack the images horizontally
stacked_rotation = np.hstack([
    np.array(cropped_image),
    np.array(rotated_90),
    np.array(rotated_180),
    np.array(rotated_270)
])

# Save the rotated image stack
rotated_image_stack = Image.fromarray(stacked_rotation)
rotated_image_stack.save('10_rotation.png')
