import cv2
import numpy as np
from scipy import ndimage
import itertools
import skimage
from skimage.transform import warp, AffineTransform
import random
import utils
from PIL import Image, ImageEnhance
import math

# ___________________ Image Convertions_________________________


def convert_to_gray(img, expand=False):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(image, axis=-1)


def convert_to_rgbg(img, equalize=False):
    gray = convert_to_gray(img)
    if equalize:
        gray = adaptive_histogram_equaluization(gray)
    return np.concatenate((img, gray), axis=2)


def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def gaussian_smoothing(img):
    return cv2.GaussianBlur(img, (3, 3), 0)


def adaptive_gaussian_thresholding(img):
    image = gaussian_smoothing(img)
    image = cv2.adaptiveThreshold(image,
                                  255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 3, 2)
    return np.expand_dims(image, axis=-1)


def adaptive_otsu(img):
    # Otsu's thresholding after Gaussian filtering
    blur = gaussian_smoothing(img)
    ret3, th3 = cv2.threshold(blur,
                              0,
                              255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3


def histogram_equaluization(img):
    image = cv2.equalizeHist(img)
    return np.expand_dims(image, axis=-1)


def adaptive_histogram_equaluization(img):
    #     image = img#gaussian_smoothing(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    image = clahe.apply(img)
    return np.expand_dims(image, axis=-1)


def sharpen_image(img):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    image = cv2.filter2D(img, -1, kernel_sharpening)
    return np.expand_dims(image, axis=-1) if img.shape[2] == 1 else image

# __________________Applying normatilzation our images__________________


def get_images_mean(imgs):
    dims = 1 if not imgs.ndim else imgs.ndim - 1
    global_mean = np.mean(imgs, axis=tuple(range(dims)))
    global_mean_image = np.mean(imgs, axis=0).astype(np.uint8)

    return global_mean, global_mean_image


def get_images_std(imgs):
    dims = 1 if not imgs.ndim else imgs.ndim - 1
    global_std = np.std(imgs, axis=tuple(range(dims)))
    return global_std


def normalize_images(imgs, mean=128., std=128.):
    # Empty array that will hold the normalized images
    normalized_images = []
    for img in imgs:
        normalized_images.append(normalize_image(img, mean, std))
    return normalized_images


def normalize_image(img, mean, std):
    return (img - mean) / std

# ___________________Image Augmentation____________________


def get_required_augmentation_for_labels(X_data, y_data, threshold=800):
    # This gets the number of requiered augmentations for each label below the threshold
    train_data_counter = utils.get_data_count(X_data, y_data)
    # Threshold for labels representaion
    requiered_augmentation = {}
    for label_no in list(set(y_data)):
        count = train_data_counter[label_no]
        if threshold > (count):  # if count is far from the threshold
            requiered_augmentation[label_no] = threshold - count  # Augment to reach the threshold
    return requiered_augmentation

# required_augmentation = get_required_augmentations_for_labels(X_train, y_train, labels_mapping)


def apply_relative_augmentation_on(X_data, y_data, threshold=1400):
    X_augmented = list(X_data)
    y_augmented = list(y_data)
    # Augmentations required by each element
    required_augmentation = get_required_augmentation_for_labels(X_augmented,
                                                                 y_augmented,
                                                                 threshold=threshold)

    for image, label in zip(X_data, y_data):
        # For signs that have a count thats below the mean
        if label in required_augmentation:
            if required_augmentation[label] > 0:
                image_augmentation = apply_augmentation(image, apply_all=True)
                X_augmented += image_augmentation
                y_augmented += len(image_augmentation) * [label]
                required_augmentation[label] -= len(image_augmentation)
    return X_augmented, y_augmented


def apply_augmentation(img, blur=False,
                       rotate=False,
                       fade=False,
                       skew=False,
                       apply_all=False):

    augmentation_results = []
    if blur or apply_all:
        augmentation_results += blur_image(img)
        # augmentation_results.append(blur_image(img))
        # print('Augmentations after blur {}'.format(len(augmentation_results)))
    if skew or apply_all:
        augmentation_results += augment_image_pov(img)
        # augmentation_results.append(augment_image_pov(img))
        # print('Augmentations after pov {}'.format(len(augmentation_results)))
        # augmention_results.append([skew_image(img), skew_image(img)])
    if rotate or apply_all:
        augmentation_results += rotate_image(img)
        # augmentation_results.append(rotate_image(img))
        # print('Augmentations after rotate {}'.format(len(augmentation_results)))
    if apply_all:
        augmentation_results += bright_dim_image(img)
        # augmentation_results.append(bright_dim_image(img))
        # print('Augmentations after brightness {}'.format(len(augmentation_results)))
    if fade or apply_all:
        augmentation_results += fade_image_color(img)
        # augmentation_results.append(fade_image_color(img))
        # print('Augmentations after fade {}'.format(len(augmentation_results)))
    return augmentation_results  # list(itertools.chain.from_iterable(augmention_results))


def blur_image(img):
    image = cv2.GaussianBlur(img, (5, 5), 0)
    if img.shape[2] == 1:
        return [np.expand_dims(image, axis=-1)]
    else:
        return [image]


def rotate_image(img):
    image1 = ndimage.rotate(img, 15, mode='nearest', reshape=False)
    image2 = ndimage.rotate(img, -15, mode='nearest', reshape=False)
    # if img.shape[2] == 1:
    #     return [np.expand_dims(image1, axis=-1),
    #             np.expand_dims(image2, axis=-1)]
    # else:
    return [image1, image2]


def bright_dim_image(img):  # This augments lighting conditions
    #     bright_img = tf.image.adjust_brightness(img, delta = 0.2).numpy()
    bright_img = adjust_contrast_brightness(img, alpha=1.1, beta=30)
#     dim_img = tf.image.adjust_brightness(img, delta = - 0.1).numpy()
    dim_img = adjust_contrast_brightness(img, alpha=0.9, beta=-80)
    if img.shape[2] == 1:
        return [np.expand_dims(bright_img, axis=-1),
                np.expand_dims(dim_img, axis=-1)]
    else:
        return [bright_img, dim_img]


def fade_image_color(img):  # Low saturation
    # Low contrast
    image = adjust_contrast_brightness(img, alpha=0.8, beta=100)
    if img.shape[2] == 1:
        return [np.expand_dims(image, axis=-1)]
    else:
        return [image]
#     return[tf.image.adjust_contrast(img, 0.2).numpy()]

# def rotate_img(img):
#     return [ndimage.rotate(img, 15, reshape=False),
#             ndimage.rotate(img, -15, reshape=False)]


def get_brightness(img):
    dims = 1 if not imgs.ndim else imgs.ndim - 1
    image_mean = np.mean(img, axis=tuple(range(dims)))
    if image_mean > 200:
        return 'Bright'
    elif image_mean < 50:
        return 'Dim'
    else:
        return 'Neutral'


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def enhance_image(img, factor=2):
    pil_image = Image.fromarray(img.astype(
        'uint8'), 'RGB') if img.shape[2] == 3 else Image.fromarray(img.squeeze(), 'L')
    factor = factor
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_image = enhancer.enhance(factor)

    if img.shape[2] == 1:
        return np.expand_dims(np.array(enhanced_image), axis=-1)
    else:
        return np.array(enhanced_image)


def change_prespective(img, direction, change_factor=1.1):

    if img is None:
        raise ValueError('img is None')
    w = img.shape[0]
    h = img.shape[1]
    # print(img.shape[2])
    # print(img.squeeze().shape)
    sym = Image.fromarray(img.astype('uint8'),
                          'RGB') if img.shape[2] == 3 else Image.fromarray(img.squeeze(), 'L')
    # max_skew_amount = max(w, h)
    # max_skew_amount = int(math.ceil(max_skew_amount * change_factor))
    # skew_amount = random.randint(1, max_skew_amount)
    # print(skew_amount)
    # top bottom 8
    # left 10

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    # if random() < self.proba:
    if direction == 'right':
        new_points = [(y1, x1),  # Top Left
                      (y2, x1 - 10),  # Top Right
                      (y2, x2 + 10),  # Bottom Right
                      (y1, x2)]   # Bottom Left
    elif direction == 'left':
        new_points = [(y1, x1 - 10),  # Top Lef
                      (y2, x1),  # Top Right
                      (y2, x2),  # Bottom Right
                      (y1, x2 + 10)]   # Bottom Left
    elif direction == 'topdown':
        new_points = [(y1 - 8, x1),                # Top Left
                      (y2 + 8, x1),                # Top Right
                      (y2, x2),  # Bottom Right
                      (y1, x2)]  # Bottom Left
    elif direction == 'bottomup':
        new_points = [(y1, x1),                # Top Left
                      (y2, x1),                # Top Right
                      (y2 + 8, x2),  # Bottom Right
                      (y1 - 8, x2)]  # Bottom Left
    else:
        new_points = [(0, 0),
                      (w-1, 0),
                      (w-1, h-1),
                      (0, h-1)]

    from_points = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]

    coeffs = find_coeffs(new_points, from_points)
    sym = sym.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return np.expand_dims(np.array(sym), axis=-1) if img.shape[2] == 1 else np.array(sym)


def augment_image_pov(img, direction='all'):
    image_povs = []
    if direction == 'all':
        image_povs = [change_prespective(img, 'left'),
                      change_prespective(img, 'right'),
                      change_prespective(img, 'topdown'),
                      change_prespective(img, 'bottomup')]
    return image_povs

# def prespective_transform(img, tilt='right'):
#     if tilt == 'right':
#
#     else:
#


def skew_image(img, shear=None,  rotation=None):
    rotate_values = [-.1, -.3]
    shear_values = [.2, .3, .4]

    affine_shear = shear if shear else random.choice(shear_values)
    affine_rotation = rotation if rotation else random.choice(rotate_values)

    affine_transform = AffineTransform(rotation=affine_rotation,
                                       shear=affine_shear)
    # Apply transform to image data
    return skimage.img_as_ubyte(warp(img, inverse_map=affine_transform,
                                     mode='symmetric'))


def adjust_contrast_brightness(img, alpha=1.1, beta=30):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

# ______________________ Imgae Preprocessing__________________________


def preprocess(img,
               grayscale=False,
               equalize=False,
               rgbg=False,
               enhance=False):
    # image = sharpen_image(img)
    image = img
    # Before
    # if enhance:
    #     image = enhance_image(image, factor=3)
    if rgbg:
        image = convert_to_rgbg(image, equalize=equalize)
    else:
        if grayscale or equalize:
            image = convert_to_gray(image)
            if equalize:
                image = adaptive_histogram_equaluization(image)
    # After
    if enhance:
        image = enhance_image(image, factor=3)
    return image


def preprocess_images(imgs, mean=128., std=128.,
                      grayscale=False,
                      equalize=False,
                      normalize=False,
                      rgbg=False,
                      enhance=False):
    # Images to retrun
    images = []
    # Apply preprocessing
    for img in imgs:
        images.append(preprocess(img,
                                 grayscale=grayscale,
                                 equalize=equalize,
                                 rgbg=rgbg, enhance=enhance))
    # At the end of it all apply Normalizatrion to the processed images
    images = np.array(images)
    if normalize:
        images = normalize_images(images, mean, std)
    return images
