import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as filedialog

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

def sobel_operator_color(image):
    # Define the Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize output arrays
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    # Pad the image to handle border pixels
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant')

    # Apply the Sobel operator on each channel
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gradient_x[i, j, c] = np.sum(padded_image[i:i+3, j:j+3, c] * sobel_x)
                gradient_y[i, j, c] = np.sum(padded_image[i:i+3, j:j+3, c] * sobel_y)

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

'''
UI part
'''

# Create a Tkinter window
window = tk.Tk()
window.title("Image Enhancement")
window.geometry("900x1000")  

# Calculate the target size for the images (1/4 of the window size)
target_width = 980 // 3
target_height = 980 // 3

# Variable to store the loaded image
loaded_image = None
loaded_image_path = None

# Function to load a different image
def load_image():
    global loaded_image
    global loaded_image_path
    filetypes = (("PNG files", "*.png"),)
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    if filepath:
        loaded_image = cv2.imread(filepath)
        loaded_image_path = filepath
        if loaded_image is None:
            print(f"Failed to read image: {filepath}")
        else:
            print(f"Image loaded: {filepath}")
            display_loaded_image()


# Function to display the loaded image on the UI
def display_loaded_image():
    # Create a PIL Image object from the loaded image
    image = Image.fromarray(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))

    # Resize the images to the target size
    image = image.resize((target_width, target_height), Image.ANTIALIAS)

    # Create a Tkinter PhotoImage object from the PIL Image
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    loaded_image_label.configure(image=photo)
    loaded_image_label.image = photo



# Function to process the image and update the UI
def process_image():
    loaded_image_path_label.configure(text="")
    if loaded_image is None:
        print("No image loaded")
        return
    
    print('Processing ' + loaded_image_path + ', wait a few seconds please ~')

    image = loaded_image

    tmp_img = image.copy()
    # Apply Sobel operator for edge detection
    edges, edges_x, edges_y = sobel_operator(tmp_img)

    # Create PIL Image objects from the processed images
    original_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    edges_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    edges_x_image = Image.fromarray(cv2.cvtColor(edges_x, cv2.COLOR_BGR2RGB))
    edges_y_image = Image.fromarray(cv2.cvtColor(edges_y, cv2.COLOR_BGR2RGB))

    # Resize the images to the target size
    original_image = original_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_image = edges_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_x_image = edges_x_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_y_image = edges_y_image.resize((target_width, target_height), Image.ANTIALIAS)

    # Create Tkinter PhotoImage objects from the PIL Images
    original_photo = ImageTk.PhotoImage(original_image)
    edges_photo = ImageTk.PhotoImage(edges_image)
    edges_x_photo = ImageTk.PhotoImage(edges_x_image)
    edges_y_photo = ImageTk.PhotoImage(edges_y_image)

    # Update the UI with the processed images
    original_label.configure(image=original_photo)
    edges_label.configure(image=edges_photo)
    edges_x_label.configure(image=edges_x_photo)
    edges_y_label.configure(image=edges_y_photo)

    # Keep a reference to the PhotoImage objects to avoid garbage collection
    original_label.image = original_photo
    edges_label.image = edges_photo
    edges_x_label.image = edges_x_photo
    edges_y_label.image = edges_y_photo

    # Clear the UI elements
    loaded_image_label.configure(image=None)
    loaded_image_label.image = None
    loaded_image_path_label.configure(text="")

    window.update()

# Function to process the image and update the UI
def process_image_color():
    loaded_image_path_label.configure(text="")
    if loaded_image is None:
        print("No image loaded")
        return
    
    print('Processing ' + loaded_image_path + ', wait a few seconds please ~')

    image = loaded_image

    tmp_img = image.copy()
    # Apply Sobel operator for edge detection
    edges, edges_x, edges_y = sobel_operator_color(tmp_img)

    # Create PIL Image objects from the processed images
    original_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    edges_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    edges_x_image = Image.fromarray(cv2.cvtColor(edges_x, cv2.COLOR_BGR2RGB))
    edges_y_image = Image.fromarray(cv2.cvtColor(edges_y, cv2.COLOR_BGR2RGB))

    # Resize the images to the target size
    original_image = original_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_image = edges_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_x_image = edges_x_image.resize((target_width, target_height), Image.ANTIALIAS)
    edges_y_image = edges_y_image.resize((target_width, target_height), Image.ANTIALIAS)

    # Create Tkinter PhotoImage objects from the PIL Images
    original_photo = ImageTk.PhotoImage(original_image)
    edges_photo = ImageTk.PhotoImage(edges_image)
    edges_x_photo = ImageTk.PhotoImage(edges_x_image)
    edges_y_photo = ImageTk.PhotoImage(edges_y_image)

    # Update the UI with the processed images
    original_label.configure(image=original_photo)
    edges_label.configure(image=edges_photo)
    edges_x_label.configure(image=edges_x_photo)
    edges_y_label.configure(image=edges_y_photo)

    # Keep a reference to the PhotoImage objects to avoid garbage collection
    original_label.image = original_photo
    edges_label.image = edges_photo
    edges_x_label.image = edges_x_photo
    edges_y_label.image = edges_y_photo

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

edges_label = tk.Label(window)
edges_label.grid(row=0, column=1, padx=10, pady=10)

edges_label_text = tk.Label(window, text="|Gx|+|Gy| Image")
edges_label_text.grid(row=1, column=1, padx=10, pady=10)

edges_x_label = tk.Label(window)
edges_x_label.grid(row=2, column=0, padx=10, pady=10)

edges_x_label_text = tk.Label(window, text="|Gx| Image")
edges_x_label_text.grid(row=3, column=0, padx=10, pady=10)

edges_y_label = tk.Label(window)
edges_y_label.grid(row=2, column=1, padx=10, pady=10)

edges_y_label_text = tk.Label(window, text="|Gy| Image")
edges_y_label_text.grid(row=3, column=1, padx=10, pady=10)

# Create a button to process the image
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.grid(row=5, column=0, columnspan=2, pady=10)

# Create a button to process the image
process_color_button = tk.Button(window, text="Process Image with color", command=process_image_color)
process_color_button.grid(row=6, column=0, columnspan=2, pady=10)

# Create a button to load a different image
load_button = tk.Button(window, text="Load Image", command=load_image)
load_button.grid(row=4, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
window.mainloop()