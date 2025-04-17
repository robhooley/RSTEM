import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw


def create_visualizer(data, user_function):
    #"""
    #Creates a Tkinter application for visualizing a 4D numpy array.

    #Parameters:
     #   data (numpy.ndarray): A 4D numpy array.
    #    user_function (function): A user-defined function that processes the data.
    #"""
    if data.ndim != 4:
        raise ValueError("Input data must be a 4D numpy array.")

    # Process data with the user function to create the main image
    main_image = user_function(data)
    main_image_pil = Image.fromarray(main_image.astype(np.uint8))

    clicked_positions = []  # Store positions of clicks
    image_refs = {"main_image": None, "current_sub_image_pil": None, "current_sub_image_tk": None}  # Prevent garbage collection

    def resize_image(image, max_dimension):
        #"""
        #Resizes an image to fit within the specified maximum dimension, preserving aspect ratio.
        #Uses nearest-neighbor interpolation to keep the image pixelated.
        #"""
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            # Fit by width
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Fit by height
            new_height = max_dimension
            new_width = int(new_height * aspect_ratio)

        return image.resize((new_width, new_height), Image.NEAREST)

    def ensure_rgb(image):
        #"""
        #Ensures the image is in RGB format. If grayscale, converts it to RGB.
        #"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def normalize_to_8bit(image):
        #"""
        #Normalizes a numpy array to 8-bit for display purposes with percentile-based clipping.
        #Ensures proper scaling even for outliers.
        #"""
        lower_percentile, upper_percentile = np.percentile(image, [1, 99])  # Get 1st and 99th percentiles
        clipped_image = np.clip(image, lower_percentile, upper_percentile)  # Clip to the range
        image_min = clipped_image.min()
        image_max = clipped_image.max()
        epsilon = 1e-5  # Prevent divide-by-zero for nearly constant images
        normalized = (255 * (clipped_image - image_min) / (image_max - image_min + epsilon)).astype(np.uint8)
        return normalized

    def update_main_image():
        #"""
        #Updates the main image displayed in the left window, applying markers to a resized version.
        #Ensures the markers are centered on the corresponding pixel and the image is displayed in RGB format.
        #"""
        # Resize the main image with nearest-neighbor interpolation
        resized_main = resize_image(main_image_pil, 1024)

        # Ensure the image is in RGB format
        resized_main = ensure_rgb(resized_main)
        resized_main = resized_main.rotate(90, Image.NEAREST, expand = 1)
        #resized_main = cv2.rotate(resized_main,cv2.ROTATE_90_CLOCKWISE)
        # Draw markers on the resized image
        draw = ImageDraw.Draw(resized_main)
        scale_factor_x = resized_main.width / main_image_pil.width
        scale_factor_y = resized_main.height / main_image_pil.height
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Scale marker positions to match resized image and adjust to center of pixel
            scaled_x = int((x + 0.5) * scale_factor_x)
            scaled_y = int((y + 0.5) * scale_factor_y)
            # Draw cross
            draw.line((scaled_x - 3, scaled_y, scaled_x + 3, scaled_y), fill="red", width=1)
            draw.line((scaled_x, scaled_y - 3, scaled_x, scaled_y + 3), fill="red", width=1)
            # Draw number
            draw.text((scaled_x + 5, scaled_y - 5), str(i), fill="red")

        # Convert to PhotoImage for display
        main_image_tk = ImageTk.PhotoImage(resized_main)

        image_refs["main_image"] = main_image_tk

        # Dynamically resize the canvas to match the image dimensions
        main_canvas.config(width=resized_main.width, height=resized_main.height)
        main_canvas.create_image(
            0, 0,  # Position at the top-left corner
            anchor=tk.NW,
            image=main_image_tk,
        )

    def update_pointer_image(x, y):
        #"""
        #Updates the right window's image based on the pointer position.
        #"""
        if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
            sub_image = data[x, y]
            normalized = normalize_to_8bit(sub_image)  # Normalize for display
            current_sub_image_pil = Image.fromarray(normalized).convert("RGB")  # Ensure color display
            image_refs["current_sub_image_pil"] = current_sub_image_pil

            # Resize the pointer image with nearest-neighbor interpolation
            resized_pointer = resize_image(
                current_sub_image_pil, min(pointer_canvas.winfo_width(), pointer_canvas.winfo_height())
            )
            pointer_image_tk = ImageTk.PhotoImage(resized_pointer)
            image_refs["current_sub_image_tk"] = pointer_image_tk
            pointer_canvas.create_image(
                pointer_canvas.winfo_width() // 2,
                pointer_canvas.winfo_height() // 2,
                anchor=tk.CENTER,
                image=pointer_image_tk,
            )

    def on_click(event):
        #"""
        #Handles click events on the main image.
        #"""
        x = int(event.x * main_image_pil.width / main_canvas.winfo_width())
        y = int(event.y * main_image_pil.height / main_canvas.winfo_height())
        clicked_positions.append((x, y))  # Store the clicked position
        update_main_image()  # Update the main image to include the new marker

    def on_right_click(event):
        #"""
        #Handles right-click events to remove the last marker.
        #"""
        if clicked_positions:
            clicked_positions.pop()  # Remove the last position
            update_main_image()  # Update the display to remove the marker


    # Tkinter GUI setup
    root = tk.Tk()
    root.title("4D Array Visualizer")

    # Determine the required initial size
    main_image_size = 1024  # Main image will be resized to 1024px
    pointer_image_size = 512  # Example size for the pointer image
    padding = 50  # Add padding for labels and buttons
    initial_width = main_image_size + pointer_image_size + padding
    initial_height = max(main_image_size, pointer_image_size) + padding

    # Set the initial size of the window
    root.geometry(f"{initial_width}x{initial_height}")
    root.minsize(initial_width, initial_height)  # Prevent resizing smaller than this

    # Create resizable layout
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    # Create left window
    left_frame = tk.Frame(root)
    left_frame.grid(row=0, column=0, sticky="nsew")
    left_label = tk.Label(left_frame, text="Main Image")
    left_label.pack()
    main_canvas = tk.Canvas(left_frame, bg="white")  # Dynamically sized by update_main_image
    main_canvas.pack(fill=tk.BOTH, expand=True)
    main_canvas.bind("<Button-1>", on_click)  # Bind click to add markers
    main_canvas.bind("<Button-3>", on_right_click)

    # Create right window
    right_frame = tk.Frame(root)
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_label = tk.Label(right_frame, text="Pointer Sub-Image")
    right_label.pack()
    pointer_canvas = tk.Canvas(right_frame, width=pointer_image_size, height=pointer_image_size, bg="white")
    pointer_canvas.pack(fill=tk.BOTH, expand=True)

    # Bind mouse motion to update pointer image
    def on_mouse_motion(event):
        x = int(event.x * main_image_pil.width / main_canvas.winfo_width())
        y = int(event.y * main_image_pil.height / main_canvas.winfo_height())
        update_pointer_image(x, y)

    main_canvas.bind("<Motion>", on_mouse_motion)

    # Display the main image initially
    update_main_image()

    # Run the Tkinter event loop
    root.mainloop()

# Example usage:
if __name__ == "__main__":
    # Example 4D array and user function
    dataset = np.load(r"C:\Users\robert.hooley\Documents\Coding\Coding_old\dataset.npy")
    


    def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
        if mask_center_coordinates is None:  # use the middle of the image
            mask_center_coordinates = (int(image_width / 2), int(image_height / 2))
        if mask_radius is None:  # use the smallest distance between the center and image walls
            mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1],
                              image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
        Y, X = np.ogrid[:image_height, :image_width]
        dist_from_center = np.sqrt((X - mask_center_coordinates[0]) ** 2 + (Y - mask_center_coordinates[1]) ** 2)
        mask = dist_from_center <= mask_radius
        return mask


    def rough_VBF(image_array):
        camera_data_shape = image_array[0][0].shape  # shape of first image to get image dimensions
        dataset_shape = image_array.shape[0], image_array.shape[1]  # scanned region shape
        radius = 60  # pixels for rough VBF image construction
        VBF_intensity_list = []  # empty list to take virtual bright field image sigals
        integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_radius=radius)
        for row in image_array:  # iterates through array rows
            for pixel in row:  # in each row iterates through pixels
                VBF_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
                VBF_intensity_list.append(VBF_intensity)  # adds to the list

        VBF_intensity_array = np.asarray(VBF_intensity_list)  # converts list to array
        VBF_intensity_array = np.reshape(VBF_intensity_array, (
        dataset_shape[0], dataset_shape[1]))  # reshapes array to match image dimensions
        return VBF_intensity_array

    create_visualizer(dataset, rough_VBF)
