from PIL import Image

# Load the image
image = Image.open('hokiebird.jpg')

# Convert to grayscale
grayscale_image = image.convert('L')
grayscale_image.save('04_grayscale.png')
