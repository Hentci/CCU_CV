import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
共12個part, 其中part6和part12各包含了16張圖
1. 原始Lena_image
2. 原始Lena_histogram
3. global approach Lena_image
4. global approach Lena histogram
5. local approach Lena_image
6. local approach Lena histogram (block * 16)
7. 原始Peppers_image
8. 原始Peppers_histogram
9. global approach Peppers_image
10. global approach Peppers histogram
11. local approach Peppers_image
12. local approach Peppers histogram (block * 16)
'''

Lena = 'HW1_test_image/Lena.bmp'
Peppers = 'HW1_test_image/Peppers.bmp'



def show_img(path):
    # Load images
    img = cv2.imread(path)

    # Display
    cv2.imshow(path, img)
    cv2.waitKey(0)

def show_new_img(img, title):
    # Display
    cv2.imshow(title, img)
    cv2.waitKey(0)

def show_histogram(data, title):
    plt.hist(data, bins=256)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()


def histogram_equalization(path):
    img = cv2.imread(path)
    [height, width, channel] = img.shape

    # sum up
    original_histogram_cnt = np.zeros(256)
    for i in range(0, height):
        for j in range(0, width):
            pixel = img[i][j][0]
            original_histogram_cnt[pixel] += 1

    # get cdf
    cdf = np.zeros(256)
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + original_histogram_cnt[i]

    # count probability
    prob = np.zeros(256)
    new_histogram = np.zeros(256)
    # total_pixels = height * width
    for i in range(0, 256):
        prob[i] = (cdf[i] - cdf.min())/ (cdf.max() - cdf.min())
        # prob[i] = cdf[i] / total_pixels
        new_histogram[i] = round(prob[i] * 255)

    # store new values
    new_img = img
    for i in range(0, height):
        for j in range(0, width):
            value = img[i][j][0]
            new_img[i][j] = new_histogram[value]

    return original_histogram_cnt, new_histogram, new_img


Lena_original_histogram, Lena_new_histogram, Lena_new_img = histogram_equalization(Lena)

print('1. 原始Lena_image')
show_img(Lena)
print('2. 原始Lena_histogram')
show_histogram(Lena_original_histogram, 'Lena_original_histogram')
print('3. global approach Lena_image')
show_histogram(Lena_new_histogram, 'Lena_new_histogram')
print('4. global approach Lena histogram')
show_new_img(Lena_new_img, 'Lena_new_img')