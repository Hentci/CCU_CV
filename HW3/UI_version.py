import math
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as filedialog

# Avoid pixel values over the range
def clamp_pixel_value(x):
    return max(0, min(x, 255))

# Use histogram equalization to enhance image
def histogram_equalization(image, channel, status, max_value=256):
    height = 0
    width = 0
    img = image.copy()
    # implement RGB case
    if status == 1:
        height, width, channels = image.shape
    else:
    # implement HSI or LAB case (preprocessing)
        height = len(image)
        width = len(image[0])
        for i in range(height):
            for j in range(width):
                # HSI * 255 -> recover to 0 - 255 
                if status == 2:
                    img[i][j][channel] = int(img[i][j][channel] * 255)
                    img[i][j][channel] = clamp_pixel_value(img[i][j][channel])
                else:
                    img[i][j][channel] = int(img[i][j][channel])

    # get cdf
    prefix_sum = [0 for _ in range(max_value)]
    for i in range(height):
        for j in range(width):
            prefix_sum[img[i][j][channel]] = prefix_sum[img[i][j][channel]] + 1

    for i in range(1, max_value):
        prefix_sum[i] = prefix_sum[i - 1] + prefix_sum[i]

    total_size = height * width
    for i in range(height):
        for j in range(width):
            value = int(round(prefix_sum[int(img[i][j][channel])] / total_size * (max_value - 1)))
            if status == 2:  # HSI -> transfer back to 0 - 1
                img[i][j][channel] = float(value / 255)
            else:
                img[i][j][channel] = value

    return img

# Convert from RGB to HSI
def convert_RGB_to_HSI(image):
    [height, width, channels] = image.shape
    hsi_image = [[[0.0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            [B, G, R] = image[i][j]
            [H, S, I] = [0, 0, 0]
            R /= 255.0
            G /= 255.0
            B /= 255.0
            # Calculate theta
            molecular = 0.5 * (R - G + R - B)
            denominator = math.sqrt((R - G) ** 2 + (R - B) * (G - B))
            if denominator == 0:  # Avoid division by 0
                theta = 0
            else:
                theta = math.acos(round(molecular / denominator, 6))
            # H value
            if B <= G:
                H = theta
            else:
                H = 2 * math.pi - theta
            # S value
            if R != 0 or G != 0 or B != 0:
                S = 1 - 3 / (R + G + B) * min(R, G, B)
            # I value
            I = (R + G + B) / 3.0
            hsi_image[i][j] = [H, S, I]
    return hsi_image

# Convert from HSI to RGB
def convert_HSI_to_RGB(image):
    [height, width] = [len(image), len(image[0])]
    rgb_image = [[[0.0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            [H, S, I] = image[i][j]
            [R, G, B] = [0.0, 0.0, 0.0]
            # RG sector (0° <= H < 120°)
            if H >= 0 and H < 2 * math.pi / 3:
                B = I * (1 - S)
                R = I * (1 + (S * math.cos(H) / math.cos(math.pi / 3 - H)))
                G = 3.0 * I - (R + B)
            # GB sector (120° <= H < 240°)
            elif H >= 2 * math.pi / 3 and H < 4 * math.pi / 3:
                H = H - 2 * math.pi / 3
                R = I * (1 - S)
                G = I * (1 + (S * math.cos(H) / math.cos(math.pi / 3 - H)))
                B = 3.0 * I - (R + G)
            # BR sector (240° <= H < 360°)
            elif H >= 4 * math.pi / 3 and H <= 2 * math.pi:
                H = H - 4 * math.pi / 3
                G = I * (1 - S)
                B = I * (1 + (S * math.cos(H) / math.cos(math.pi / 3 - H)))
                R = 3 * I - (G + B)
            rgb_image[i][j] = [clamp_pixel_value(B * 255), clamp_pixel_value(G * 255), clamp_pixel_value(R * 255)]
    return rgb_image

# RGB to LAB function
def h(q):
    if q > 0.008856:
        return pow(q, 1 / 3)
    else:
        return 7.787 * q + 16 / 116

# Convert from RGB to LAB
def convert_RGB_to_LAB(image):
    [height, width, channels] = image.shape
    lab_image = [[[0.0 for _ in range(3)] for _ in range(width)] for _ in range(height)]

    for i in range(height):
        for j in range(width):
            [B, G, R] = image[i][j]
            [L, a, b] = [0.0, 0.0, 0.0]
            R /= 255.0
            G /= 255.0
            B /= 255.0
            Xn = 0.95045
            Yn = 1.00000
            Zn = 1.08875
            # Convert to CIE XYZ
            X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
            Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
            Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B
            # Convert to CIE Lab
            X = X / Xn
            Y = Y / Yn
            Z = Z / Zn
            L = 116 * h(Y) - 16
            a = 500 * (h(X) - h(Y))
            b = 200 * (h(Y) - h(Z))
            lab_image[i][j] = [L, a, b]
    return lab_image

# LAB to RGB function
def f(t):
    if t > 0.008856:
        return pow(t, 3)
    else:
        return 3 * (0.008865) ** 2 * (t - (16 / 116))

# Convert from LAB to RGB
def convert_LAB_to_RGB(image):
    [height, width] = [len(image), len(image[0])]
    rgb_image = [[[0.0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            [L, a, b] = image[i][j]
            [R, G, B] = [0.0, 0.0, 0.0]
            Xn = 0.95045
            Yn = 1.00000
            Zn = 1.08875
            # Convert to CIE XYZ
            X = Xn * f(1 / 116 * (L + 16) + 1 / 500 * a)
            Y = Yn * f(1 / 116 * (L + 16))
            Z = Zn * f(1 / 116 * (L + 16) - 1 / 200 * b)
            # Convert to RGB
            R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
            G = -0.969266 * X + 1.8760108 * Y + 0.0415560 * Z
            B = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
            rgb_image[i][j] = [clamp_pixel_value(B * 255), clamp_pixel_value(G * 255), clamp_pixel_value(R * 255)]
    return rgb_image

'''
UI part
'''

# Create a Tkinter window
window = tk.Tk()
window.title("Image Enhancement")
window.geometry("850x850")  

# Calculate the target size for the images (1/4 of the window size)
target_width = 1200 // 3
target_height = 900 // 3

# Variable to store the loaded image
loaded_image = None
loaded_image_path = None

# Function to load a different image
def load_image():
    global loaded_image
    global loaded_image_path
    filetypes = (("JPEG files", "*.jpg"), ("PNG files", "*.png"))
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    if filepath:
        loaded_image = cv.imread(filepath)
        loaded_image_path = filepath
        if loaded_image is None:
            print(f"Failed to read image: {filepath}")
        else:
            print(f"Image loaded: {filepath}")
            display_loaded_image()

# Function to display the loaded image on the UI
def display_loaded_image():
    # Create a PIL Image object from the loaded image
    image = Image.fromarray(cv.cvtColor(loaded_image, cv.COLOR_BGR2RGB))

    # Resize the images to the target size
    image = image.resize((target_width, target_height), Image.ANTIALIAS)

    # Create a Tkinter PhotoImage object from the PIL Image
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    loaded_image_label.configure(image=photo)
    loaded_image_label.image = photo

    # Display the loaded image path
    # loaded_image_path_label.configure(text="Loaded Image: " + loaded_image_path)


# Function to process the image and update the UI
def process_image():
    loaded_image_path_label.configure(text="")
    if loaded_image is None:
        print("No image loaded")
        return
    
    print('Processing ' + loaded_image_path + ', wait a few seconds please ~')

    image = loaded_image
    image_RGB = image.copy()
    image_HSI = convert_RGB_to_HSI(image)
    image_LAB = convert_RGB_to_LAB(image)

    # COLOR ENHANCEMENT
    result_RGB = histogram_equalization(histogram_equalization(histogram_equalization(image_RGB, 0, 1), 1, 1), 2, 1)
    tmp_HSI = convert_HSI_to_RGB(histogram_equalization(image_HSI, 2, 2)) # only process I channel
    tmp_LAB = convert_LAB_to_RGB(histogram_equalization(image_LAB, 0, 3, 101)) # L max vale = 100, only process L channel

    # BACK TO IMAGE FORM
    result_HSI = image.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(3):
                result_HSI[i][j][k] = tmp_HSI[i][j][k]

    result_LAB = image.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(3):
                result_LAB[i][j][k] = tmp_LAB[i][j][k]

    # Create PIL Image objects from the processed images
    original_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    rgb_image = Image.fromarray(cv.cvtColor(result_RGB, cv.COLOR_BGR2RGB))
    hsi_image = Image.fromarray(cv.cvtColor(result_HSI, cv.COLOR_BGR2RGB))
    lab_image = Image.fromarray(cv.cvtColor(result_LAB, cv.COLOR_BGR2RGB))

    # Resize the images to the target size
    original_image = original_image.resize((target_width, target_height), Image.ANTIALIAS)
    rgb_image = rgb_image.resize((target_width, target_height), Image.ANTIALIAS)
    hsi_image = hsi_image.resize((target_width, target_height), Image.ANTIALIAS)
    lab_image = lab_image.resize((target_width, target_height), Image.ANTIALIAS)

    # Create Tkinter PhotoImage objects from the PIL Images
    original_photo = ImageTk.PhotoImage(original_image)
    rgb_photo = ImageTk.PhotoImage(rgb_image)
    hsi_photo = ImageTk.PhotoImage(hsi_image)
    lab_photo = ImageTk.PhotoImage(lab_image)

    # Update the UI with the processed images
    original_label.configure(image=original_photo)
    rgb_label.configure(image=rgb_photo)
    hsi_label.configure(image=hsi_photo)
    lab_label.configure(image=lab_photo)

    # Keep a reference to the PhotoImage objects to avoid garbage collection
    original_label.image = original_photo
    rgb_label.image = rgb_photo
    hsi_label.image = hsi_photo
    lab_label.image = lab_photo

    # Clear the UI elements
    loaded_image_label.configure(image=None)
    loaded_image_label.image = None
    loaded_image_path_label.configure(text="")

    window.update()


# Create labels to display the images and loaded image path
loaded_image_label = tk.Label(window)
loaded_image_label.grid(row=0, column=0, padx=10, pady=10)

loaded_image_path_label = tk.Label(window)
loaded_image_path_label.grid(row=1, column=0, padx=10)

# Create labels to display the images
original_label = tk.Label(window)
original_label.grid(row=0, column=0, padx=10, pady=10)

original_label_text = tk.Label(window, text="Original Image")
original_label_text.grid(row=1, column=0, padx=10, pady=10)

rgb_label = tk.Label(window)
rgb_label.grid(row=0, column=1, padx=10, pady=10)

rgb_label_text = tk.Label(window, text="RGB Image")
rgb_label_text.grid(row=1, column=1, padx=10, pady=10)

hsi_label = tk.Label(window)
hsi_label.grid(row=2, column=0, padx=10, pady=10)

hsi_label_text = tk.Label(window, text="HSI Image")
hsi_label_text.grid(row=3, column=0, padx=10, pady=10)

lab_label = tk.Label(window)
lab_label.grid(row=2, column=1, padx=10, pady=10)

lab_label_text = tk.Label(window, text="LAB Image")
lab_label_text.grid(row=3, column=1, padx=10, pady=10)

# Create a button to process the image
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.grid(row=5, column=0, columnspan=2, pady=10)

# Create a button to load a different image
load_button = tk.Button(window, text="Load Image", command=load_image)
load_button.grid(row=4, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
window.mainloop()


