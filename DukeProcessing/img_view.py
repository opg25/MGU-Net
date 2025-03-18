import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
image_path = 'DukeProcessing/DukeData/train/images/Subject_02_00.npy'
mask_path = 'DukeProcessing/DukeData/train/masks/Subject_02_00.npy'

image = np.load(image_path)
mask = np.load(mask_path)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the image
ax1.imshow(image, cmap='gray')
ax1.set_title('Image')
ax1.axis('off')

# Plot the mask
ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')
ax2.axis('off')

# Print shape information
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")

plt.tight_layout()
plt.show()

