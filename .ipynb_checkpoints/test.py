import matplotlib.pyplot as plt
from skimage import feature, transform, io
from skimage.color import rgb2gray
from skimage.feature import plot_matches, ORB
from skimage.measure import ransac
from skimage.util import img_as_ubyte
import numpy as np
import math


""" On charge deux images et on les fusionne en mode panoramique. On utilise 
    ORB pour trouver les points d'intérêts et RANSAC pour les aligner.
"""

def trim_image(image, x_offset=0) :
    mask = image == 0
    rows = np.nonzero((~mask).sum(axis=1))
    cols = np.nonzero((~mask).sum(axis=0))
    trimmed = image[rows[0].min():rows[0].max()+1, cols[0].min():cols[0].max()-x_offset,:]
    return trimmed


plt.rcParams['figure.figsize'] = (10, 10)

# TODO: Lecture de plusieurs images
img_1 = io.imread("images/small_image1.jpg")
img_2 = io.imread("images/small_image2.jpg")


""" On trouve les points d'intérêts """
orb = ORB()
orb.detect_and_extract(rgb2gray(img_1))
keypoints_1 = orb.keypoints
descriptors_1 = orb.descriptors

orb.detect_and_extract(rgb2gray(img_2))
keypoints_2 = orb.keypoints
descriptors_2 = orb.descriptors


""" On trouve les correspondances """
matches = feature.match_descriptors(descriptors_1, descriptors_2, cross_check=True)

if keypoints_1 is None or keypoints_2 is None:
    print("No keypoints found")
    exit(0)

src = keypoints_1[matches[:, 0]][:, ::-1]
dst = keypoints_2[matches[:, 1]][:, ::-1]


""" On aligne les images """
model_robust, inliers = ransac((src, dst), transform.AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)

# La rotation retournée ne correspond pas toujours à la rotation appliquée
# par transform.warp ci-dessous, laissant une partie du triangle noir à
# l'image. Doublage de la valeur par sécurité.
x_offset = 2 * int(abs(img_2.shape[0] * math.tan(model_robust.rotation)))
outliers = inliers == False
print(f'Rotation : {math.degrees(model_robust.rotation)} deg')
print(f'{x_offset} px to delete on X axis')


""" On met la deuxième image à sa place """
# TODO: Taille de l'image de sortie
img_2_warped = img_as_ubyte(transform.warp(img_2, model_robust, output_shape=(img_1.shape[0], img_1.shape[1] + img_2.shape[1])))


""" On merge les deux images """
# TODO: Blending
result = np.copy(img_2_warped)
result[:img_1.shape[0], :img_1.shape[1]] = img_1
result = trim_image(result, x_offset)


""" Affichage """
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].imshow(img_1)
ax[0][0].set_title('Image 1')

ax[0][1].imshow(img_2_warped)
ax[0][1].set_title('Image 2')

plot_matches(ax[1][0], img_1, img_2, keypoints_1, keypoints_2, matches)
ax[1][0].set_title('Correspondances')

ax[1][1].imshow(result)
ax[1][1].set_title('Résultat')

plt.show()
