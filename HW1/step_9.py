from PIL import Image
import numpy as np

# Load the grayscale image
image = Image.open('hokiebird.jpg').convert('L')
image_np = np.array(image)

# Create the negative
negative_image_np = 255 - image_np

# Save the negative image
negative_image = Image.fromarray(negative_image_np)
negative_image.save('09_negative.png')
