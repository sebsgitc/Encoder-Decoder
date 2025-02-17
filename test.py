import imageio.v3 as iio
import matplotlib.pyplot as plt

# Load the image
image_path = './images/r01_0211.rec.16bit.tif'
image = iio.imread(image_path)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(image.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of r01_0211.rec.16bit.tif')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

