from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from PCV.tools import imtools, pca

# Get list of images and their size
imlist = imtools.get_imlist('../data/fontimages/')  # fontimages.zip is part of the book data set
im = np.array(Image.open(imlist[0]))  # open one image to get the size
m, n = im.shape[:2]

# Create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(imname)).flatten() for imname in imlist], 'f')

# Perform PCA
V, S, immean = pca.pca(immatrix)

# Show the images (mean and 7 first modes)
# This gives figure 1-8 (p15) in the book.
plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m, n))
for i in range(7):
    plt.subplot(2, 4, i + 2)
    plt.imshow(V[i].reshape(m, n))
plt.show()
