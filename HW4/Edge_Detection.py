import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_operator(image):
    # Convert image to grayscale
    gray = image.mean(axis=2)

    ''' kernel = 3 x 3 '''
    # Define the Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    ''' kernel = 5 x 5 '''
    # sobel_x = np.array([[-1, -2, 0, 2, 1],
    #                 [-4, -8, 0, 8, 4],
    #                 [-6, -12, 0, 12, 6],
    #                 [-4, -8, 0, 8, 4],
    #                 [-1, -2, 0, 2, 1]])

    # sobel_y = np.array([[-1, -4, -6, -4, -1],
    #                 [-2, -8, -12, -8, -2],
    #                 [0, 0, 0, 0, 0],
    #                 [2, 8, 12, 8, 2],
    #                 [1, 4, 6, 4, 1]])

    ''' kernel = 10 x 10 '''
    # sobel_x = np.array([[-1, -2, -3, -4, -5, 0, 5, 4, 3, 2],
    #                     [-2, -4, -6, -8, -10, 0, 10, 8, 6, 4],
    #                     [-3, -6, -9, -12, -15, 0, 15, 12, 9, 6],
    #                     [-4, -8, -12, -16, -20, 0, 20, 16, 12, 8],
    #                     [-5, -10, -15, -20, -25, 0, 25, 20, 15, 10],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [5, 10, 15, 20, 25, 0, -25, -20, -15, -10],
    #                     [4, 8, 12, 16, 20, 0, -20, -16, -12, -8],
    #                     [3, 6, 9, 12, 15, 0, -15, -12, -9, -6],
    #                     [2, 4, 6, 8, 10, 0, -10, -8, -6, -4]])

    # sobel_y = np.array([[-1, -2, -3, -4, -5, 0, 5, 4, 3, 2],
    #                     [-2, -4, -6, -8, -10, 0, 10, 8, 6, 4],
    #                     [-3, -6, -9, -12, -15, 0, 15, 12, 9, 6],
    #                     [-4, -8, -12, -16, -20, 0, 20, 16, 12, 8],
    #                     [-5, -10, -15, -20, -25, 0, 25, 20, 15, 10],
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                     [5, 10, 15, 20, 25, 0, -25, -20, -15, -10],
    #                     [4, 8, 12, 16, 20, 0, -20, -16, -12, -8],
    #                     [3, 6, 9, 12, 15, 0, -15, -12, -9, -6],
    #                     [2, 4, 6, 8, 10, 0, -10, -8, -6, -4]])

    # Pad the image to handle border pixels
    padded_image = np.pad(gray, ((1, 1), (1, 1)), mode='constant')

    # Initialize output arrays
    gradient_x = np.zeros_like(gray, dtype=np.float32)
    gradient_y = np.zeros_like(gray, dtype=np.float32)

    # Apply the Sobel operator
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gradient_x[i, j] = np.sum(padded_image[i:i+3, j:j+3] * sobel_x)
            gradient_y[i, j] = np.sum(padded_image[i:i+3, j:j+3] * sobel_y)

    # Compute the magnitude of the gradients
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)

    # Normalize the gradient magnitude to 0-255
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_x = (gradient_x / np.max(gradient_x)) * 255
    gradient_y = (gradient_y / np.max(gradient_y)) * 255

    # Convert the gradient magnitude to uint8 format
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    gradient_x = gradient_x.astype(np.uint8)
    gradient_y = gradient_y.astype(np.uint8)

    return gradient_magnitude, gradient_x, gradient_y



files = ['baboon.png', 'peppers.png', 'pool.png']

for file in files:

    # Load the color images
    image = cv2.imread(file)

    # Convert BGR to RGB format using array slicing
    image = image[:, :, ::-1]
    
    tmp_img = image.copy()
    # Apply Sobel operator for edge detection
    edges, edges_x, edges_y = sobel_operator(tmp_img)


    '''   4張圖合在一起, 如果要看大圖, 註解這邊, 解註解最下面cv2部份    '''
    # Create a figure and subplots
    fig, ax1 = plt.subplots(2, 2, figsize=(10, 10))

    # Display the images and labels
    ax1[0, 0].imshow(image)
    ax1[0, 0].set_title('Original')
    ax1[0, 1].imshow(edges, cmap='gray')
    ax1[0, 1].set_title('|Gx| + |Gy| Edges')
    ax1[1, 0].imshow(edges_x, cmap='gray')
    ax1[1, 0].set_title('|Gx| Edges')
    ax1[1, 1].imshow(edges_y, cmap='gray')
    ax1[1, 1].set_title('|Gy| Edges')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the combined image and labels
    plt.show()


    ''' 大圖 '''
    # # Display the original and edge images (按Esc切下一張圖)
    # image = image[:, :, ::-1]
    # cv2.imshow('Original', image)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Edges_X', edges_x)
    # cv2.imshow('Edges_Y',  edges_y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()