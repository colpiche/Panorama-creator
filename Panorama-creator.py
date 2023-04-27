import matplotlib.pyplot as plt
from skimage import feature, transform, io
from skimage.color import rgb2gray
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.util import img_as_ubyte
import numpy as np
import math

plt.rcParams['figure.figsize'] = (10, 10)


""" On charge deux images et on les fusionne en mode panoramique. On utilise
    ORB pour trouver les points d'intérêts et RANSAC pour les aligner.
"""


def trim_image(image, x_offset=0):
    mask = image == 0
    rows = np.nonzero((~mask).sum(axis=1))
    cols = np.nonzero((~mask).sum(axis=0))
    trimmed = image[rows[0].min():rows[0].max()+1,
                    cols[0].min():cols[0].max()-x_offset,
                    :]
    return trimmed


def merge_images(img_1, img_2, keypoints_2=None, descriptors_2=None):
    """ Fusionne deux images en mode panoramique """

    orb = ORB()
    orb.detect_and_extract(rgb2gray(img_1))
    keypoints_1 = orb.keypoints
    descriptors_1 = orb.descriptors

    if keypoints_2 is None or descriptors_2 is None:
        print("Keypoints non-transmis.")
        orb.detect_and_extract(rgb2gray(img_2))
        keypoints_2 = orb.keypoints
        descriptors_2 = orb.descriptors

    """ On trouve les correspondances """
    matches = feature.match_descriptors(descriptors_1,
                                        descriptors_2,
                                        cross_check=True)

    if keypoints_1 is None or keypoints_2 is None:
        print("No keypoints found")
        exit(0)

    src = keypoints_1[matches[:, 0]][:, ::-1]
    dst = keypoints_2[matches[:, 1]][:, ::-1]

    """ On aligne les images """
    model_robust, inliers = ransac((src, dst),
                                   transform.AffineTransform,
                                   min_samples=3,
                                   residual_threshold=2,
                                   max_trials=100)

    # La rotation retournée ne correspond pas toujours à la rotation appliquée
    # par transform.warp ci-dessous, laissant une partie du triangle noir à
    # l'image. Doublage de la valeur par sécurité.
    x_offset = 2 * int(abs(img_2.shape[0] * math.tan(model_robust.rotation)))
    print(f'Rotation : {math.degrees(model_robust.rotation)} deg')
    print(f'{x_offset} px to delete on X axis')

    """ On met la deuxième image à sa place """
    img_2_warped = img_as_ubyte(
        transform.warp(img_2,
                       model_robust,
                       output_shape=(img_1.shape[0],
                                     img_1.shape[1] + img_2.shape[1])))

    """ On merge les deux images """
    # TODO: Blending
    result = np.copy(img_2_warped)
    result[:img_1.shape[0], :img_1.shape[1]] = img_1
    result = trim_image(result, x_offset)

    return result, keypoints_1, descriptors_1


if __name__ == "__main__":
    kp = None
    des = None

    l_images = [
        "images/small_image1.jpg",
        "images/small_image2.jpg",
        "images/small_image3.jpg",
        "images/small_image4.jpg",
        "images/small_image5.jpg"
    ]

    result = io.imread(l_images[-1])

    for i in range(1, len(l_images)):
        img_name = l_images[len(l_images)-i-1]
        print(f'Merging {img_name}')
        result, kp, des = merge_images(io.imread(img_name), result, kp, des)

    """ Affichage """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(result)
    ax.set_title('Résultat')

    plt.show()
