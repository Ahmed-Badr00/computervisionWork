from PIL import Image
import numpy as np

# Load the image
image = Image.open('hokiebird.jpg')
image_np = np.array(image)

# Compute the average over the R, G, B channels
average_image_np = np.mean(image_np, axis=2).astype(np.uint8)

# Save the averaged image
average_image = Image.fromarray(average_image_np)
average_image.save('05_average.png')
