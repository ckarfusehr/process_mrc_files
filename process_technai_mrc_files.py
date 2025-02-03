#### Importing packages ####

import os
import tkinter as tk
from tkinter import ttk, filedialog
import concurrent.futures  # for multi-threading
import re
import numpy as np
from skimage import exposure
from scipy.ndimage import zoom
import mrcfile
import tifffile
from PIL import Image, ImageDraw, ImageFont
import sys

#### Constants ####

MAG_TO_PIXEL_DICT = {
    40: 260.46,
    50: 208.368,
    60: 173.64,
    80: 130.3,
    100: 108.97,
    120: 93.2,
    145: 74.3,
    190: 53.7,
    240: 45.31,
    260: 45.31,
    290: 37.88,
    380: 28.77,
    470: 23.05,
    560: 19.45,
    1100: 9.734,
    1650: 6.519,
    2100: 5.071,
    2700: 3.824,
    3200: 3.354,
    4400: 2.45,
    6500: 1.644,
    11000: 0.9945,
    15000: 0.7096,
    21000: 0.5129,
    26000: 0.4215,
    30000: 0.3653,
    42000: 0.261,
    52000: 0.211,
    67000: 0.163,
    110000: 0.102,
}

PIXEL_LENGTH_TO_SCALEBAR_LENGTH_IN_nm_DICT = {
    260.46: 50000,
    208.368: 50000,
    173.64: 10000,
    130.3: 10000,
    108.97: 10000,
    93.2: 10000,
    74.3: 10000,
    53.7: 10000,
    45.31: 10000,
    37.88: 100,
    28.77: 10000,
    23.05: 10000,
    19.45: 10000,
    9.734: 5000,
    6.519: 5000,
    5.071: 5000,
    3.824: 1000,
    3.354: 1000,
    2.45: 1000,
    1.644: 1000,
    0.9945: 1000,
    0.7096: 500,
    0.5129: 1000,
    0.4215: 200,
    0.3653: 200,
    0.261: 200,
    0.211: 100,
    0.163: 100,
    0.102: 100,
}

SPLIT_OUTPUT_TO_SINGLE_FILES = False

#### Backend functions ####

def read_magnifications_from_mdoc(mrc_file_path):
    """
    Reads the magnification values from a .mdoc file corresponding to the given .mrc file.

    :param mrc_file_path: str, path to the .mrc file
    :return: list of int, magnification values extracted from the .mdoc file
    """

    base_path, extension = os.path.splitext(mrc_file_path)
    possible_mdoc_paths = [
        base_path + extension + ".mdoc",
        base_path + ".mdoc"
    ]

    magnifications = []
    mdoc_found = False
    for mdoc_file_path in possible_mdoc_paths:
        if os.path.exists(mdoc_file_path):
            mdoc_found = True
            with open(mdoc_file_path, "r") as mdoc_file:
                for line in mdoc_file:
                    match = re.search(r"Magnification = (\d+)", line)
                    if match:
                        magnifications.append(int(match.group(1)))
            break

    if not mdoc_found:
        raise FileNotFoundError("No .mdoc file found corresponding to the .mrc file.")

    return magnifications

def map_magnifications_to_pixel_size(magnifications):
    """
    Maps the magnification values to pixel sizes using the predefined MAG_TO_PIXEL_DICT.

    :param magnifications: list of int, magnification values
    :return: list of float, corresponding pixel sizes in nanometers
    """
    pixel_sizes = []
    for mag in magnifications:
        if mag in MAG_TO_PIXEL_DICT:
            pixel_sizes.append(MAG_TO_PIXEL_DICT[mag])
        else:
            raise KeyError(f"Magnification {mag} not found in MAG_TO_PIXEL_DICT.")
    return pixel_sizes

def rescale_intensity_to_16bit(image):
    """
    Rescales the intensity of the input image to 16-bit.

    :param image: numpy array, input image
    :return: numpy array, rescaled image with intensity values in the range of 0 to 65535 (16-bit)
    """

    return exposure.rescale_intensity(image, out_range=(0, 65535)).astype(np.uint16)

def resize_image(image, target_shape):
    """
    Resizes the input image to the specified target shape using the scipy.ndimage.zoom function.

    :param image: numpy array, input image
    :param target_shape: tuple, target shape in the form (height, width)
    :return: numpy array, resized image with the specified target shape
    """

    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=1)

def draw_rectangle(image, x1, y1, x2, y2, fill_value):
    """
    Draws a filled rectangle on the input image with the specified coordinates and fill value.

    :param image: numpy array, input image
    :param x1: int, x-coordinate of the top-left corner of the rectangle
    :param y1: int, y-coordinate of the top-left corner of the rectangle
    :param x2: int, x-coordinate of the bottom-right corner of the rectangle
    :param y2: int, y-coordinate of the bottom-right corner of the rectangle
    :param fill_value: int, fill value for the rectangle
    :return: numpy array, image with the rectangle drawn
    """

    image[int(y1):int(y2), int(x1):int(x2)] = fill_value
    return image

def draw_text(image, text, x, y, font_size):
    """
    Draws the specified text on the input image at the given (x, y) coordinates using the provided font size.

    :param image: numpy array, input image
    :param text: str, text to be drawn on the image
    :param x: int, x-coordinate of the text position
    :param y: int, y-coordinate of the text position
    :param font_size: int, font size to be used for the text
    :return: numpy array, image with the text drawn
    """
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=0)
    return np.array(image_pil)

def calculate_scalebar_parameters(img_height, img_width, pixel_length, position=None):
    """
    Calculates the parameters necessary to draw a scale bar on an image.

    :param img_height: int, the height of the input image in pixels.
    :param img_width: int, the width of the input image in pixels.
    :param pixel_length: float, the length of one pixel in nanometers.
    :param position: tuple, optional, a tuple containing the (x, y) coordinates of the top-left corner of the scale bar.
                     If None, the scale bar is placed in the lower-right corner of the image.
    :return: tuple, a tuple containing the following scale bar parameters:
             - scale_thickness (int): the thickness of the scale bar in pixels.
             - scale_length_in_nm (float): the length of the scale bar in nanometers.
             - x1 (int): the x-coordinate of the top-left corner of the scale bar.
             - y1 (int): the y-coordinate of the top-left corner of the scale bar.
             - x2 (int): the x-coordinate of the bottom-right corner of the scale bar.
             - y2 (int): the y-coordinate of the bottom-right corner of the scale bar.
    """

    # Define the scalebar parameters
    scale_thickness = img_height // 125
    scale_length_in_nm = PIXEL_LENGTH_TO_SCALEBAR_LENGTH_IN_nm_DICT[pixel_length]
    scale_length_in_px = int(round(scale_length_in_nm / pixel_length))

    # Determine the scalebar position and dimensions
    if position is None:
        x_buffer = img_width // 30
        y_buffer = img_height // 30
        x1 = img_width - x_buffer - scale_length_in_px
        y1 = img_height - y_buffer - scale_thickness
    else:
        x1, y1 = position
    x2 = x1 + scale_length_in_px
    y2 = y1 + scale_thickness

    return scale_thickness, scale_length_in_nm, x1, y1, x2, y2

def draw_scalebar(image, pixel_length, position=None):
    """
    Draws a scale bar and corresponding annotations on the input image based on the provided pixel length.

    :param image: numpy array, input image
    :param pixel_length: float, pixel length in nanometers
    :param position: tuple, optional, x and y coordinates of the scale bar's starting position; if not provided,
                     the scale bar will be placed in the lower left corner of the image
    :return: numpy array, image with the scale bar and annotations drawn
    """

    img_height, img_width = image.shape
    scale_thickness, scale_length_in_nm, x1, y1, x2, y2 = calculate_scalebar_parameters(
        img_height, img_width, pixel_length, position
    )

    # Draw the scalebar and text on the image
    image = draw_rectangle(image, x1 - 30, y1 - 60, x2 + 30, y2 + 60, 65535)  # Frame
    image = draw_rectangle(image, x1, y1, x2, y2, 0)

    font_size = img_height // 90

    length_scale = "nm"
    if scale_length_in_nm >= 1000:
        scale_length_in_nm /= 1000
        length_scale = "µm"

    # Add the scale length to the image
    text = f"{scale_length_in_nm} {length_scale}"
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text_width, text_height = font.getsize(text)
    text_x = (x1 + x2) / 2 - text_width / 2
    text_y = y1 - text_height - scale_thickness / 2
    image = draw_text(image, text, text_x, text_y, font_size)

    # Add the pixel length to the image
    text = f"{pixel_length} {length_scale}/px"
    text_width, text_height = font.getsize(text)
    text_y = y1 + scale_thickness
    image = draw_text(image, text, text_x, text_y, font_size)

    return image

def combine_images_horizontally(images):
    """
    Combines a list of images horizontally, assuming all input images have the same shape.

    :param images: list of numpy arrays, input images
    :return: numpy array, combined image
    """

    if not all(img.shape == images[0].shape for img in images[1:]):
        raise ValueError("All images should have the same size.")
    return np.concatenate(images, axis=1)

def process_single_image(image, pixel_size):
    """
    Processes a single image for equalization, rescaling, and scalebar drawing.

    :param image: numpy array, input image
    :param pixel_size: float, pixel size of the image in nanometers
    :return: numpy array, processed image with added scalebar and annotations
    """

    original = image
    equalize_adapthist = exposure.equalize_adapthist(image, clip_limit=0.01, nbins=256)

    original_rescaled = rescale_intensity_to_16bit(original)
    equalize_adapthist_rescaled = rescale_intensity_to_16bit(equalize_adapthist)

    original_rescaled = draw_scalebar(original_rescaled, pixel_size)
    equalize_adapthist_rescaled = draw_scalebar(equalize_adapthist_rescaled, pixel_size)

    images = [original_rescaled, equalize_adapthist_rescaled]
    combined_image = combine_images_horizontally(images)

    return combined_image

def read_mrc_file_to_array(mrc_file_path):
    """
    Reads an MRC file and returns its data as a numpy array.

    :param mrc_file_path: str, path to the MRC file
    :return: numpy array, the MRC data array
    """

    with mrcfile.open(mrc_file_path, mode="r", permissive=True) as mrc:
        mrc_data = np.array(mrc.data)

    return mrc_data

def multithread_process_single_images(mrc_data_array, pixel_sizes):
    """
    Processes multiple images in parallel, each image with its corresponding pixel size.

    :param mrc_data_array: numpy array, input MRC data array
    :param pixel_sizes: list of float, list of pixel sizes in nanometers
    :return: numpy array, array of processed images
    """
    if len(mrc_data_array) != len(pixel_sizes):
        raise ValueError("Number of images and number of pixel sizes do not match.")

    processed_images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_single_image, mrc_data_array, pixel_sizes)
        for result in results:
            processed_images.append(result)

    return np.array(processed_images)

def process_mrc_file_main(mrc_file_path):
    """
    Processes an MRC file containing multiple images, adds scale bars and annotations, and saves the result as a TIFF file.

    :param mrc_file_path: str, path to the input MRC file
    """

    mrc_data_array = read_mrc_file_to_array(mrc_file_path)
    magnifications = read_magnifications_from_mdoc(mrc_file_path)
    pixel_sizes = map_magnifications_to_pixel_size(magnifications)

    if len(mrc_data_array) != len(pixel_sizes):
        raise ValueError("Number of images in MRC file does not match number of pixel sizes.")

    processed_images_arr = multithread_process_single_images(mrc_data_array, pixel_sizes)

    base_path, extension = os.path.splitext(mrc_file_path)

    if SPLIT_OUTPUT_TO_SINGLE_FILES is False:
        output_tiff_path = base_path + "_processed.tiff"
        tifffile.imwrite(
            output_tiff_path,
            processed_images_arr,
            photometric="minisblack",
            metadata={"axes": "IYX"},
        )
    else:
        output_tiff_path = base_path + "_processed"
        for i in range(len(processed_images_arr)):
            tifffile.imwrite(
                output_tiff_path + "_" + str(i) + ".tiff",
                processed_images_arr[i],
                photometric="minisblack",
            )

def start_gui():
    root = tk.Tk()
    root.geometry("600x200")
    app = Application(master=root)
    app.mainloop()

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Technai File Processor")
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Define new style for orange button
        self.style.configure("Orange.TButton", foreground="black", background="orange")

        # Define new style for green button
        self.style.configure("Green.TButton", foreground="black", background="green")

        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        self.selected_files_listbox = tk.Listbox(self, height=10)
        self.selected_files_listbox.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_files)
        self.browse_button.pack(side="top", padx=10, pady=10)

        # Set initial button style to green
        self.process_button = ttk.Button(
            self, text="Process .mrc", command=self.process_files, style="Green.TButton"
        )
        self.process_button.pack(side="top", padx=10, pady=10)

    def browse_files(self):
        file_path = filedialog.askopenfilename(filetypes=[("MRC files", "*.mrc")])
        if file_path:
            self.selected_files_listbox.insert(tk.END, file_path)

    def process_files(self):

        for index in range(self.selected_files_listbox.size()):
            file_path = self.selected_files_listbox.get(index)
            try:
                process_mrc_file_main(file_path)
                print(f"Finished processing {file_path}")
            except Exception as e:
                print(f"Failed processing {file_path}: {e}")

        # Change button color back to green
        self.process_button.configure(style="Green.TButton")
        self.process_button.update()

        self.selected_files_listbox.delete(0, tk.END)

    def quit(self):
        self.master.destroy()

def cmd_wrapper():
    # remove the first argument, which is the name of the script itself
    args = sys.argv[1:]

    for options in args:
        if options == "-s":
            print(
                "-s option is activated. Processed MRC stack files will be saved as individual TIFF images."
            )
            global SPLIT_OUTPUT_TO_SINGLE_FILES
            SPLIT_OUTPUT_TO_SINGLE_FILES = True

        else:
            mrc_file_path = options
            try:
                print(f"Started processing {mrc_file_path}")
                process_mrc_file_main(mrc_file_path)
                print(f"Finished processing {mrc_file_path}")
            except Exception as e:
                print(f"Failed processing {mrc_file_path}: {e}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        start_gui()
    else:
        cmd_wrapper()

