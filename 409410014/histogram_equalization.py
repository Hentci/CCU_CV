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

Lena_path = 'HW1_test_image/Lena.bmp'
Peppers_path = 'HW1_test_image/Peppers.bmp'
Lena = cv2.imread(Lena_path)
Peppers = cv2.imread(Peppers_path)


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
    plt.hist(data, bins=256, range=(0, 255))
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

def show_all_local_block_histogram(block_histo):
    # Define number of blocks and plot layout
    num_blocks = 16
    num_rows = int(np.sqrt(num_blocks))
    num_cols = int(np.ceil(num_blocks / num_rows))

    # Create figure and axis objects
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # Iterate over blocks and plot histograms
    for i in range(num_blocks):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].hist(block_histo[i], bins=256, range=(0, 255))
        axs[row_idx, col_idx].set_title(f"Block {i+1}")
        axs[row_idx, col_idx].set_xlim([0, 255])

    # Add overall title and axis labels
    fig.suptitle("Local Histograms")
    fig.text(0.5, 0.04, 'Pixel Value', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

    # Display plot
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close('all')


# global approach
def histogram_equalization(img):
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

# local approach
def local_approach(img):
    res = img
    # will contain 16 histogram
    blocks_histo = []
    [height, width, channel] = img.shape
    block_height = height / 4
    block_width = width / 4
    for i in range(0, 4):
        for j in range(0, 4):
            x = i * block_height
            y = j * block_width
            # comfirm the block range from original image
            block = img[int(x) : int(x + block_height - 1), int(y) : int(y + block_width - 1)]
            # local histogram equalization
            original_histo, new_histo, res[int(x) : int(x + block_height - 1), int(y) : int(y + block_width - 1)] = histogram_equalization(block)
            blocks_histo.append(new_histo)

    return blocks_histo, res


'''
Lena: part1 ~ part6, part6需要等待一下, 因為有16張圖
'''
Lena_original_histogram, Lena_new_histogram, Lena_new_img = histogram_equalization(Lena)
print('1. 原始Lena_image')
show_img(Lena_path)
print('2. 原始Lena_histogram')
show_histogram(Lena_original_histogram, 'Lena_original_histogram')
print('3. global approach Lena_image')
show_new_img(Lena_new_img, 'Lena_new_img')
print('4. global approach Lena histogram')
show_histogram(Lena_new_histogram, 'Lena_new_histogram')

Lena_local_approach_histogram, Lena_local_approach_img = local_approach(Lena)

print('5. local approach Lena_image')
show_new_img(Lena_local_approach_img, 'Lena_local_approach_img')

print('6. local approach Lena histogram (block * 16)')
show_all_local_block_histogram(Lena_local_approach_histogram)
cv2.destroyAllWindows()

'''
Peppers: part7 ~ part12, part12需要等待一下, 因為有16張圖
'''
Peppers_original_histogram, Peppers_new_histogram, Peppers_new_img = histogram_equalization(Peppers)
print('7. 原始Peppers_image')
show_img(Peppers_path)
print('8. 原始Peppers_histogram')
show_histogram(Peppers_original_histogram, 'Peppers_original_histogram')
print('9. global approach Peppers_image')
show_new_img(Peppers_new_img, 'Peppers_new_img')
print('10. global approach Peppers histogram')
show_histogram(Peppers_new_histogram, 'Peppers_new_histogram')

Peppers_local_approach_histogram, Peppers_local_approach_img = local_approach(Peppers)

print('11. local approach Peppers_image')
show_new_img(Peppers_local_approach_img, 'Peppers_local_approach_img')

print('12. local approach Peppers histogram (block * 16)')
show_all_local_block_histogram(Peppers_local_approach_histogram)
cv2.destroyAllWindows()