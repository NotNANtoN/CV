import imageio
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb, rgb2gray
import numpy as np

im = imageio.imread('Elbphilharmonie.jpg')
# k-means
im_gray = rgb2gray(im)

# Flatten the image to obtain list of points
im_gray_flat = im_gray.flatten()

# parameters
k = 100
change_threshold = 0.0001 # tolerance for changes in cluster centers to detect changes

# 1. randomly choose cluster centers
centers = np.random.choice(im_gray_flat, size=k)

# we want to use NumPy broadcasting in the loop below, so let's reshape the arrays to be able to do that
im_gray_flat = im_gray_flat.reshape((im_gray_flat.size, 1)) # to [N, 1]
centers = centers.reshape((1,k)) # to [1, k]


changed = True
while changed:
	# 2. calculate cluster assignments
	dist = np.abs(im_gray_flat - centers) # result is [N, k] thanks to broadcasting
	assignments = np.argmin(dist, axis=1)
	
	# 3. update cluster centers, see if there are changes
	new_centers = np.zeros((1,k))
	changed = False
	for i in range(k):
		assigned_to_cluster_i = assignments == i # boolean array
		new_centers[0,i] = np.average(im_gray_flat[assigned_to_cluster_i])
		if np.abs(new_centers[0,i] - centers[0,i]) > change_threshold:
			changed = True

		# assign new centers
		centers = new_centers

# Visualization
assignments_img = assignments.reshape(im_gray.shape)
labels_kmeans_colored = label2rgb(assignments_img, im, kind='overlay')

fig, ax = plt.subplots(ncols=2, figsize=(12,12))
ax[0].imshow(im_gray, cmap='gray')
ax[1].imshow(labels_kmeans_colored)
plt.savefig('kmeans_elbphi.png', bbox_inches='tight')
plt.show()

