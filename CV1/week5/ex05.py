import numpy as np
from scipy.signal import convolve2d
import imageio
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.filters import sobel
from skimage.filters import laplace
from skimage.filters.rank import minimum, maximum
import matplotlib.pyplot as plt

# 1. Preparation
I = imageio.imread('woman.png')
N = random_noise(I, mode='gaussian', var=0.01)
S = gaussian(I, sigma=1.0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(N, cmap='gray')
ax[0].set_title('Noisy')
ax[1].imshow(S, cmap='gray')
ax[1].set_title('Smoothed')
plt.show()


# 2. Sobel
Fn = sobel(N)
Fs = sobel(S)


tn = 0.15
ts = 0.05

Bn = Fn > tn
Bs = Fs > ts

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Fn, cmap='gray')
ax[0,0].set_title('Sobel noisy')
ax[1,0].imshow(Fs, cmap='gray')
ax[1,0].set_title('Sobel smoothed')
ax[0,1].imshow(Bn, cmap='gray')
ax[0,1].set_title('Mask noisy')
ax[1,1].imshow(Bs, cmap='gray')
ax[1,1].set_title('Mask smoothed')
plt.show()

# 3. Laplacian
Ln = laplace(N)
Ls = laplace(S)

# Detect zero-crossings
def zero_crossing(L, threshold=0.1):
	h, w = L.shape
	B = np.zeros((h,w), dtype=np.bool) # initialize to false
	for i in range(1,h-1): # skip borders
		for j in range(1,w-1): # skip borders
			v = L[i,j] # value of this
			N = [n for n in [L[i-1,j], L[i+1,j], L[i,j-1], L[i,j+1]] if np.sign(v) != np.sign(n)]
			if len(N) > 0:
				m = np.abs(N).min()
				has_smallest_abs = np.abs(v) < m
				if has_smallest_abs:
					d = np.abs([v-n for n in N]).max()
					if d > threshold:
						B[i,j] = True
	return B

LBn = zero_crossing(Ln, threshold=0.25)
LBs = zero_crossing(Ls, threshold=0.02)


fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Ln, cmap='gray')
ax[0,0].set_title('Laplace noisy')
ax[1,0].imshow(Ls, cmap='gray')
ax[1,0].set_title('Laplace smoothed')
ax[0,1].imshow(LBn, cmap='gray')
ax[0,1].set_title('Mask noisy')
ax[1,1].imshow(LBs, cmap='gray')
ax[1,1].set_title('Mask smoothed')
plt.show()
