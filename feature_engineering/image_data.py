def flip_horizontal(img):
    import numpy as np
    return np.fliplr(img)


def flip_vertical(img):
    import numpy as np
    return np.flipud(img)


def rotate_right(img, rot_angle=45):
    from skimage.transform import rotate
    return rotate(img, angle=rot_angle)


def rotate_left(img, rot_angle=45):
    from skimage.transform import rotate
    return rotate(img, angle=-rot_angle)


def shift_right(img, shift_pixels=100):
    from skimage.transform import AffineTransform, warp
    transform = AffineTransform(translation=(-shift_pixels, 0))
    return warp(img, transform, mode="wrap")


def shift_left(img, shift_pixels=100):
    from skimage.transform import AffineTransform, warp
    transform = AffineTransform(translation=(shift_pixels, 0))
    return warp(img, transform, mode="wrap")


def shift_up(img, shift_pixels=100):
    from skimage.transform import AffineTransform, warp
    transform = AffineTransform(translation=(0, -shift_pixels))
    return warp(img, transform, mode="wrap")


def shift_down(img, shift_pixels=100):
    from skimage.transform import AffineTransform, warp
    transform = AffineTransform(translation=(0, shift_pixels))
    return warp(img, transform, mode="wrap")


def add_noise(img):
    from skimage.util import random_noise
    return random_noise(img)


def blur_image(img, blur_filter_size=5):
    import cv2
    return cv2.GaussianBlur(img, (blur_filter_size, blur_filter_size), 0)


def augment_image(img, rot_angle=45, shift_pixels=100, blur_filter_size=5):
    return [
        flip_horizontal(img),  # flipped horizontally
        flip_vertical(img),  # flipped vertically
        rotate_right(img, rot_angle),  # rotated clockwise
        rotate_left(img, rot_angle),  # rotated anti-clockwise
        shift_right(img, shift_pixels),  # shifted right by n pixels
        shift_left(img, shift_pixels),  # shifted left by n pixels
        shift_up(img, shift_pixels),  # shifted up by n pixels
        shift_down(img, shift_pixels),  # shifted down by n pixels
        add_noise(img),  # Random noise added
        blur_image(img, blur_filter_size),  # Blurred by a filter of selected size
    ]
