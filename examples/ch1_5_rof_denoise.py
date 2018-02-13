import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
from scipy.misc import imsave

from PCV.tools import rof

"""
This is the de-noising example using ROF in Section 1.5.
"""

# create synthetic image with noise
im = np.zeros((500, 500))
im[100:400, 100:400] = 128
im[200:300, 200:300] = 255
im += 30 * np.random.standard_normal((500, 500))

U, T = rof.denoise(im, im)
G = filters.gaussian_filter(im, 10)

# save the result
imsave('synth_original.pdf', im)
imsave('synth_rof.pdf', U)
imsave('synth_gaussian.pdf', G)

# plot
plt.figure()
plt.gray()

plt.subplot(1, 3, 1)
plt.imshow(im)
plt.axis('equal')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(G)
plt.axis('equal')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(U)
plt.axis('equal')
plt.axis('off')

plt.show()
