import cv2

# read image by cv2
original_moon = cv2.imread("HW2_test_image/blurry_moon.tif")
original_skeleton = cv2.imread("HW2_test_image/skeleton_orig.bmp")

# Define dx and dy variables for the Laplacian operator
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

# Define a function to standardize the value and keep it within 0-255 range
def standardize(value):
    return max(0, min(value, 255))


def Laplacian(img):
    # Create a new image with the same shape as the input image
    lap_img = img.copy()
    # Get the height, width, color channels of the input image
    height, width, channel = img.shape
    # Loop through every pixel in the image
    for i in range(height):
        for j in range(width):
            cur_sum = 0
            # Loop through the four neighbors of the current pixel
            for k in range(4):
                x = i + dx[k]
                y = j + dy[k]
                # Check if the neighbor is within the image boundaries
                if 0 <= x < height and 0 <= y < width:
                    # Subtract the neighbor's value from the current pixel's value
                    cur_sum -= img[x, y, 0]
            # Add 5 times the current pixel's value to the sum
            cur_sum += 5 * img[i, j, 0]
            # standardize the sum to keep it within 0-255 range
            cur_sum = standardize(cur_sum)
            # Set the pixel value in the new image to the standardizeed sum
            lap_img[i, j] = [cur_sum] * 3

    return lap_img

def High_Boost(img, A):
    new_img = img.copy()
    height, width, channel = img.shape
    # Loop through every pixel in the image
    for i in range(height):
        for j in range(width):
            cur_sum = 0
            # Loop through the four neighbors of the current pixel
            for k in range(4):
                x = i + dx[k]
                y = j + dy[k]
                # Check if the neighbor is within the image boundaries
                if 0 <= x < height and 0 <= y < width:
                    # Subtract the neighbor's value from the current pixel's value
                    cur_sum -= img[x, y, 0]
            # Add (4 + A) times the current pixel's value to the sum
            cur_sum += (4 + A) * img[i, j, 0]
            # standardize the sum to keep it within 0-255 range
            cur_sum = standardize(cur_sum)
            # Set the pixel value in the new image to the standardizeed sum
            new_img[i, j] = [cur_sum] * 3
    # Return the new image
    return new_img

# press esc to next img

# show original moon
cv2.imshow('original_moon', original_moon)
cv2.waitKey(0)

# show laplacian moon
cv2.imshow('Laplacian_moon', Laplacian(original_moon))
cv2.waitKey(0)

# show high-boost with A = 1.2 moon
cv2.imshow('high-boost with A = 1.2 moon', High_Boost(original_moon, 1.2))
cv2.waitKey(0)

# show high-boost with A = 1.7 moon
cv2.imshow('high-boost with A = 1.7 moon', High_Boost(original_moon, 1.7))
cv2.waitKey(0)

# show original skeleton
cv2.imshow('original_skeleton', original_skeleton)
cv2.waitKey(0)

# show laplacian skeleton
cv2.imshow('Laplacian_skeleton', Laplacian(original_skeleton))
cv2.waitKey(0)

# show high-boost with A = 1.2 skeleton
cv2.imshow('high-boost with A = 1.2 skeleton', High_Boost(original_skeleton, 1.2))
cv2.waitKey(0)

# show high-boost with A = 1.7 skeleton
cv2.imshow('high-boost with A = 1.7 skeleton', High_Boost(original_skeleton, 1.7))
cv2.waitKey(0)

cv2.destroyAllWindows()