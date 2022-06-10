import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib


# show as plt
def imshow(img, cmap=None):
    if cmap == "bgr":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


# show in separate window
def imshow_cv2(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale_img(img, scale):
    img_height = img.shape[0]
    img_width = img.shape[1]
    new_height = int(scale * img_height)
    new_width = int(scale * img_width)
    return cv2.resize(img, (new_width, new_height))


def check_out_of_frame(x1, x2, y1, y2, img, img_small=None):
    height = img.shape[0]
    width = img.shape[1]

    x1_new = max(x1, 0)
    x2_new = min(x2, width)
    y1_new = max(y1, 0)
    y2_new = min(y2, height)

    if img_small is not None:
        height_sm = img_small.shape[0]
        width_sm = img_small.shape[1]
        x1_sm = x1_new - x1
        x2_sm = width_sm - (x2 - x2_new)
        y1_sm = y1_new - y1
        y2_sm = height_sm - (y2 - y2_new)

        img_small = img_small[y1_sm:y2_sm, x1_sm:x2_sm]

    return x1_new, x2_new, y1_new, y2_new, img_small


def overlay_with_png(img, img_small, x1, x2, y1, y2):
    alpha_s = img_small[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        img[y1:y2, x1:x2, c] = (alpha_s * img_small[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])


def blur_border(img, x1, x2, y1, y2, left, top, right, bottom, ksize, sigma):
    height, width = img.shape[:2]
    l1, l2 = x1 - left, x1 + left
    t1, t2 = y1 - top, y1 + top
    r1, r2 = x2 - right, x2 + right
    b1, b2 = y2 - bottom, y2 + bottom
    l1, t1, r1, b1 = tuple(map(lambda x: x if x > 0 else 0, (l1, t1, r1, b1)))
    l2, r2 = tuple(map(lambda x: x if x < width else width, (l2, r2)))
    t2, b2 = tuple(map(lambda x: x if x < height else height, (t2, b2)))

    # blur left part
    img[y1:y2, l1:l2] = cv2.GaussianBlur(src=img[y1:y2, l1:l2], ksize=ksize, sigmaX=sigma)
    # blur top part
    img[t1:t2, x1:x2] = cv2.GaussianBlur(src=img[t1:t2, x1:x2], ksize=ksize, sigmaX=sigma)
    # blur right part
    img[y1:y2, r1:r2] = cv2.GaussianBlur(src=img[y1:y2, r1:r2], ksize=ksize, sigmaX=sigma)
    # blur bottom part
    img[b1:b2, x1:x2] = cv2.GaussianBlur(src=img[b1:b2, x1:x2], ksize=ksize, sigmaX=sigma)

    return img
