import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

#Load image
img = Image.open("C:\\Users\\USER\\Downloads\\coins.jpg")   
img_gray = img.convert("L")     # grayscale

gray = np.array(img_gray)

#Gaussian smoothing (noise reduction)
blur = ndimage.gaussian_filter(gray, sigma=1)

#Otsu thresholding (manual implementation)
hist, bins = np.histogram(blur.flatten(), 256, [0, 256])
hist = hist.astype(float)
hist /= hist.sum()

omega = np.cumsum(hist)
mu = np.cumsum(hist * np.arange(256))
mu_t = mu[-1]

sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
threshold = np.argmax(sigma_b2)

binary = np.zeros_like(blur)
binary[blur < threshold] = 1   # coins are darker → foreground

#Morphological operations
structure = np.ones((3, 3))

#Erosion then dilation (opening)
eroded = ndimage.binary_erosion(binary, structure=structure, iterations=2)
dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=2)

#Connected component labeling
labeled_img, num_objects = ndimage.label(dilated)

print("Number of objects detected:", num_objects)

#Display results
plt.figure(figsize=(12,8 ))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(binary, cmap="gray")
plt.title(f"Otsu Threshold (T={threshold})")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(eroded, cmap="gray")
plt.title("After Erosion")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(dilated, cmap="gray")
plt.title("After Dilation")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(labeled_img, cmap="nipy_spectral")
plt.title("Connected Components")
plt.text(
    20, 70,
    f"Objects detected: {num_objects}",
    color="yellow",
    fontsize=16,
    bbox=dict(facecolor="black", alpha=0.7)
)
plt.axis("off")

plt.tight_layout()
plt.show()