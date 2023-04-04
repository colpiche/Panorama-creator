import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import feature, transform, io
from skimage.color import rgb2gray
from skimage.feature import plot_matches, ORB
from skimage.measure import ransac
from skimage.transform import AffineTransform


""" On charge deux images et on les fusionne en mode panoramique. On utilise 
    ORB pour trouver les points d'intérêts et RANSAC pour les aligner.
"""


plt.rcParams['figure.figsize'] = (10, 10)

img_1 = rgb2gray(io.imread("images/small_image1.jpg"))
img_2 = rgb2gray(io.imread("images/small_image2.jpg"))


""" On trouve les points d'intérêts """
orb = ORB()
orb.detect_and_extract(img_1)
keypoints_1 = orb.keypoints
descriptors_1 = orb.descriptors

orb.detect_and_extract(img_2)
keypoints_2 = orb.keypoints
descriptors_2 = orb.descriptors


""" On trouve les correspondances """
matches = feature.match_descriptors(descriptors_1, descriptors_2, cross_check=True)
src = keypoints_1[matches[:, 0]][:, ::-1]
dst = keypoints_2[matches[:, 1]][:, ::-1]


""" On aligne les images """
model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)
outliers = inliers == False


""" On fusionne les images """
result = transform.warp(img_2, model_robust, output_shape=(img_1.shape[0], img_1.shape[1] + img_2.shape[1]))
result[0:img_1.shape[0], 0:img_1.shape[1]] = img_1


""" Affichage """
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].imshow(img_1, cmap="gray")
ax[0][0].set_title('Image 1')

ax[0][1].imshow(img_2, cmap="gray")
ax[0][1].set_title('Image 2')

ax[1][0].imshow(result, cmap="gray")
ax[1][0].set_title('Résultat')

plot_matches(ax[1][1], img_1, img_2, keypoints_1, keypoints_2, matches)
ax[1][1].axis('off')
ax[1][1].set_title('Correspondances')

plt.show()