from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
This example shows how images are represented using pixels, color channels and data types.
"""

# read image to array
im = np.array(Image.open('../data/empire.jpg'))
print("Shape is: {0} of type {1}".format(im.shape, im.dtype))

# read grayscale version to float array
im = np.array(Image.open('../data/empire.jpg').convert('L'), 'f')
print("Shape is: {0} of type {1}".format(im.shape, im.dtype))

# visualize the pixel value of a small region
col_1, col_2 = 190, 225
row_1, row_2 = 230, 265

# crop using array slicing
crop = im[col_1:col_2, row_1:row_2]
cols, rows = crop.shape

print("Created crop of shape: {0}".format(crop.shape))

# generate all the plots
plt.figure()
plt.imshow(im)
plt.gray()
plt.plot([row_1, row_2, row_2, row_1, row_1], [col_1, col_1, col_2, col_2, col_1], linewidth=2)
plt.axis('off')

plt.figure()
plt.imshow(crop)
plt.gray()
plt.axis('off')

plt.figure()
plt.imshow(crop)
plt.gray()
plt.plot(20 * np.ones(cols), linewidth=2)
plt.axis('off')

plt.figure()
plt.plot(crop[20, :])
plt.ylabel("Graylevel value")

from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.gca(projection='3d')
# surface plot with transparency 0.5
X, Y = np.meshgrid(np.arange(cols), -np.arange(rows))
ax.plot_surface(X, Y, crop, alpha=0.5, cstride=2, rstride=2)

plt.show()
