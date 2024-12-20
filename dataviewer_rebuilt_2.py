import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2 as cv2
import os
from expert_pi.RSTEM.analysis_functions import *
from tqdm import tqdm
import fnmatch
import tifffile as tiff
import natsort
import multiprocessing as mp


# Global variable to control mouse motion functionality
mouse_motion_enabled = True  # Initially enabled


def askfloat_helper(*args, **kwargs):
    root = tk.Tk()
    root.withdraw()
    result_queue = kwargs.pop('result_queue')
    kwargs['parent'] = root
    result_queue.put(
        simpledialog.askfloat(
            prompt="Please enter known scan width in pixels",
            title="Enter Scan Width")) #, *args, **kwargs))
    root.destroy()


def ask_float(*args, **kwargs):
    result_queue = mp.Queue()
    kwargs['result_queue'] = result_queue
    askfloat_thread = mp.Process(target=askfloat_helper, args=args, kwargs=kwargs)
    askfloat_thread.start()
    result = result_queue.get()
    if askfloat_thread.is_alive():
        askfloat_thread.join()
    return result

def create_visualizer():
    """
    Creates a Tkinter application for visualizing a 4D numpy array.
    """
    root = tk.Tk()
    root.title("4D Array Visualizer")

    # Global variables to store the data and selected function
    data_array = None
    selected_function = tk.StringVar(value="BF")
    radius_value = tk.IntVar(value=50)  # Default radius value
    circle_center = [None, None]  # Default circle center (to be dynamically set)

    # Predefined functions for processing
    functions = {
        "BF": VBF,
        "DF": VDF,
        "ADF": VADF
    }

    # Canvas and state variables
    main_image_pil = None
    pointer_image_pil = None
    clicked_positions = []  # Store positions of clicks
    image_refs = {"main_image": None, "current_sub_image": None}  # Prevent garbage collection



    def save_images():
        """
        Saves the left image (main image with markers), sub-images for clicked positions,
        and the list of clicked positions to disk.
        """
        if data_array is None or main_image_pil is None:
            print("No data or main image available to save.")
            return

        # Ask user to select a directory
        save_dir = filedialog.askdirectory()
        if not save_dir:
            print("Save canceled.")
            return

        # Save the main image with markers
        main_image_with_markers = main_image_pil.copy()
        draw = ImageDraw.Draw(main_image_with_markers)
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Draw markers on the main image
            draw.line((x - 3, y, x + 3, y), fill="red", width=1)
            draw.line((x, y - 3, x, y + 3), fill="red", width=1)
            draw.text((x + 5, y - 5), str(i), fill="red")

        # Save the main image
        main_image_path = f"{save_dir}/main_image_with_markers.tiff"
        main_image_with_markers.save(main_image_path)

        # Save individual sub-images
        for i, (x, y) in enumerate(clicked_positions, start=1):
            if 0 <= x < data_array.shape[0] and 0 <= y < data_array.shape[1]:
                sub_image = data_array[x, y]
                sub_image_path = f"{save_dir}/clicked_position_{i}_sub_image.tiff"
                Image.fromarray(sub_image).save(sub_image_path)

        # Save clicked positions to a text file
        positions_path = f"{save_dir}/clicked_positions.txt"
        with open(positions_path, "w") as f:
            for i, (x, y) in enumerate(clicked_positions, start=1):
                f.write(f"Position {i}: ({x}, {y})\n")

        print(f"Images and positions saved to {save_dir}")
        messagebox.showinfo("Save Complete", f"Images and positions saved to {save_dir}")

    def load_series():
        nonlocal data_array
        folder_path = filedialog.askdirectory(mustexist=True)

        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(
                ('.tiff', '.tif'))
        ])
        if not image_files:
            print("No image files found in the selected folder.")
            return

        num_files = len(image_files)  # counts how many .tiff files are in the directory
        guessed_scan_width = int(np.sqrt(num_files))  # assumes it is a square acquisition

        scan_width = int(ask_float(guessed_scan_width,num_files))

        # Create a progress bar and label in the main window
        progress_frame = tk.Frame(root)
        progress_frame.pack(side=tk.TOP, pady=10)

        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(side=tk.LEFT, padx=10)
        progress_bar["maximum"] = len(image_files)

        progress_label = tk.Label(progress_frame, text="0 / 0")
        progress_label.pack(side=tk.LEFT, padx=5)

        print(scan_width)
        print(folder_path)
        folder = os.listdir(folder_path)

        image_list = []


        #for file in tqdm(folder):  # iterates through folder with a progress bar
        for idx, image_file in enumerate(image_files, start=1):
            image_path = os.path.join(folder_path, image_file)
            image = tiff.imread(image_path)#Image.open(image_path)#cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # loads them with openCV
            image_list.append(image)  # adds them to a list of all images
            # Update progress bar
            progress_bar["value"] = idx
            progress_label.config(text=f"Image series import progress: {idx} / {len(image_files)} ({(idx / len(image_files)) * 100:.1f}%)")
            root.update()  # Keep the UI responsive


        array = np.asarray(image_list)  # converts the list to an array
        cam_pixels_x, cam_pixels_y = image_list[0].shape
        data_array = np.reshape(array, (scan_width, int((num_files/scan_width)), cam_pixels_x, cam_pixels_y))
        progress_frame.destroy()
        messagebox.showinfo("Import Complete", f"Successfully imported {len(image_files)} images.")
        update_function()



    def resize_image(image, max_dimension):
        """
        Resizes an image to fit within the specified maximum dimension, preserving aspect ratio.
        Uses nearest-neighbor interpolation to keep the image pixelated.
        """
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            # Fit by width
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Fit by height
            new_height = max_dimension
            new_width = int(new_height / aspect_ratio)

        return image.resize((new_width, new_height), Image.NEAREST)

    def ensure_rgb(image):
        """
        Ensures the image is in RGB format. If grayscale, converts it to RGB.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def normalize_to_8bit(image):
        """
        Normalizes a numpy array to 8-bit for display purposes with percentile-based clipping.
        """
        lower_percentile, upper_percentile = np.percentile(image, [1, 99])  # Get 1st and 99th percentiles
        clipped_image = np.clip(image, lower_percentile, upper_percentile)  # Clip to the range
        image_min = clipped_image.min()
        image_max = clipped_image.max()
        normalized = (255 * (clipped_image - image_min) / (image_max - image_min)).astype(np.uint8)
        return normalized

    def update_main_image(trigger_user_function=False):
        """
        Updates the main image displayed in the left window.
        If `trigger_user_function` is True, regenerates the main image using the user-defined function.
        """
        nonlocal main_image_pil

        # Regenerate the main image if requested
        if trigger_user_function:
            if data_array is None:
                print("No data array available.")
                return
            function = functions[selected_function.get()]
            # Call the user-defined function to regenerate the main image
            new_main_image = function(data_array, radius_value.get(), tuple(circle_center))
            normalized_main_image = normalize_to_8bit(new_main_image)
            main_image_pil = Image.fromarray(normalized_main_image).convert("RGB")

        if main_image_pil is None:
            main_canvas.delete("all")
            return

        # Resize the main image to fit the left canvas
        resized_main = main_image_pil.resize((1024, 1024), Image.NEAREST)
        resized_main = ensure_rgb(resized_main)

        # Draw markers on the resized image
        draw = ImageDraw.Draw(resized_main)
        scale_factor_x = resized_main.width / main_image_pil.width
        scale_factor_y = resized_main.height / main_image_pil.height
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Scale marker positions to match the resized image
            scaled_x = int((x + 0.5) * scale_factor_x)
            scaled_y = int((y + 0.5) * scale_factor_y)
            draw.line((scaled_x - 3, scaled_y, scaled_x + 3, scaled_y), fill="red", width=1)
            draw.line((scaled_x, scaled_y - 3, scaled_x, scaled_y + 3), fill="red", width=1)
            draw.text((scaled_x + 5, scaled_y - 5), str(i), fill="red")

        # Update the main canvas
        main_image_tk = ImageTk.PhotoImage(resized_main)

        image_refs["main_image"] = main_image_tk
        main_canvas.create_image(0, 0, anchor=tk.NW, image=main_image_tk)

    def update_pointer_image(x, y):
        """
        Updates the right window's image based on the pointer position.
        Displays a semi-transparent circle with the radius set by the user.
        """
        nonlocal pointer_image_pil, circle_center
        if data_array is None or main_image_pil is None:
            return

        if 0 <= x < data_array.shape[0] and 0 <= y < data_array.shape[1]:
            sub_image = data_array[x, y]
            normalized = normalize_to_8bit(sub_image)
            pointer_image_pil = Image.fromarray(normalized).convert("RGB")

            # Resize the pointer image to fit the square canvas
            resized_pointer = pointer_image_pil.resize((SQUARE_CANVAS_SIZE, SQUARE_CANVAS_SIZE), Image.NEAREST)

            # Scale circle_center to the resized image dimensions
            scale_x = SQUARE_CANVAS_SIZE / pointer_image_pil.width
            scale_y = SQUARE_CANVAS_SIZE / pointer_image_pil.height
            upscaled_center_x = int(circle_center[0] * scale_x)
            upscaled_center_y = int(circle_center[1] * scale_y)

            # Draw the semi-transparent circle
            draw = ImageDraw.Draw(resized_pointer, "RGBA")
            circle_radius = radius_value.get()
            draw.ellipse(
                (upscaled_center_x - circle_radius, upscaled_center_y - circle_radius,
                 upscaled_center_x + circle_radius, upscaled_center_y + circle_radius),
                outline=(255, 0, 0, 128),  # Semi-transparent red
                width=3
            )

            pointer_image_tk = ImageTk.PhotoImage(resized_pointer)
            image_refs["current_sub_image"] = pointer_image_tk
            pointer_canvas.create_image(0, 0, anchor=tk.NW, image=pointer_image_tk)

    def on_pointer_click(event):
        """
        Handles clicks on the pointer image to set the circle's center.
        Maps the mouse click to the actual pointer image coordinates.
        """
        nonlocal circle_center, pointer_image_pil
        if pointer_image_pil is None:
            return

        # Get Canvas dimensions
        canvas_width = pointer_canvas.winfo_width()
        canvas_height = pointer_canvas.winfo_height()

        # Pointer image dimensions (square)
        image_width = pointer_image_pil.width
        image_height = pointer_image_pil.height

        # The canvas is forced to be square, so drawn size matches SQUARE_CANVAS_SIZE
        drawn_width = SQUARE_CANVAS_SIZE
        drawn_height = SQUARE_CANVAS_SIZE

        # Since the canvas is square, offsets are always 0
        x_offset = 0
        y_offset = 0

        # Debugging information
        print(f"Canvas Size: {canvas_width}x{canvas_height}")
        print(f"Image Size: {image_width}x{image_height}")
        print(f"Drawn Size: {drawn_width}x{drawn_height}")
        print(f"Offsets: x={x_offset}, y={y_offset}")
        print(f"Click Position: x={event.x}, y={event.y}")

        # Scaling factors
        scale_x = image_width / drawn_width
        scale_y = image_height / drawn_height

        # Map the click position to the original image coordinates
        circle_center[0] = int((event.x - x_offset) * scale_x)
        circle_center[1] = int((event.y - y_offset) * scale_y)

        # Debugging output for circle center
        print(f"Circle Center Updated: {circle_center[0]}, {circle_center[1]}")

        # Update the pointer image to reflect the new circle center
        update_pointer_image(circle_center[0], circle_center[1])

        update_function()

    def on_click(event):
        """
        Handles left-click events on the main image to add markers.
        """
        #global clicked_positions
        if main_image_pil is None:
            return

        x = int(event.x * main_image_pil.width / main_canvas.winfo_width())
        y = int(event.y * main_image_pil.height / main_canvas.winfo_height())
        clicked_positions.append((x, y))
        update_main_image()

    def toggle_mouse_motion(event=None):
        """
        Toggles the mouse motion functionality on and off with right-click.
        """
        global mouse_motion_enabled  # Access the global variable
        mouse_motion_enabled = not mouse_motion_enabled
        state = "enabled" if mouse_motion_enabled else "suspended"
        print(f"Mouse motion functionality is now {state}.")

    def _on_mouse_motion(event):
        """
        Handles mouse motion over the main image.
        Updates the pointer image dynamically based on the cursor position.
        """
        global mouse_motion_enabled  # Access the global variable
        if not mouse_motion_enabled:  # Skip updates if mouse motion is suspended
            return

        if data_array is None:
            # If no data array is available, skip the update
            #print("No data array loaded.")
            return

        # Map the mouse position to the data array
        x = int(event.x * data_array.shape[0] / main_canvas.winfo_width())
        y = int(event.y * data_array.shape[1] / main_canvas.winfo_height())

        # Ensure indices are within bounds
        x = max(0, min(data_array.shape[0] - 1, x))
        y = max(0, min(data_array.shape[1] - 1, y))

        update_pointer_image(x, y)

    def load_data():
        """
        Loads a .npy file as a 4D data array.
        """
        nonlocal data_array
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Array", "*.npy")])
        if file_path:
            data_array = np.load(file_path)
            update_function()

    def update_function(*args):
        """
        Updates the main image based on the selected function.
        Passes both the radius and the circle center to the user function.
        """
        nonlocal main_image_pil
        if data_array is None or selected_function.get() not in functions:
            return
        function = functions[selected_function.get()]

        # Ensure the circle center is valid
        if circle_center[0] is None or circle_center[1] is None:
            circle_center[0] = 0
            circle_center[1] = 0

        # Call the user function with radius and center
        main_image = function(data_array, radius_value.get(), tuple(circle_center))
        main_image_pil = Image.fromarray(normalize_to_8bit(main_image))
        update_main_image()

    # UI Layout


    # Set the desired square canvas size
    SQUARE_CANVAS_SIZE = 512  # Example size

    # Top Frame for Controls
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    # Load Data Button
    load_button = tk.Button(top_frame, text="Load Data", command=load_data)
    load_button.pack(side=tk.LEFT, padx=5, pady=5)

    #Save images button
    save_images = tk.Button(top_frame, text="Save Images", command=save_images)
    save_images.pack(side=tk.LEFT, padx=5, pady=5)

    #Load Series button
    load_series_b = tk.Button(top_frame, text="Load image series", command=load_series)
    load_series_b.pack(side=tk.LEFT, padx=5, pady=5)


    # Radius Label and Entry
    radius_label = tk.Label(top_frame, text="Radius:")
    radius_label.pack(side=tk.LEFT, padx=5, pady=5)

    radius_entry = tk.Entry(top_frame, textvariable=radius_value, width=5)
    radius_entry.pack(side=tk.LEFT, padx=5, pady=5)
    radius_entry.bind("<Return>", update_function)

    # Function Dropdown Menu
    function_dropdown = ttk.Combobox(top_frame, textvariable=selected_function, values=list(functions.keys()),
                                     state="readonly")
    function_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
    function_dropdown.bind("<<ComboboxSelected>>", update_function)

    # Regenerate Main Image Button
    #trigger_button = tk.Button(top_frame, text="Regenerate Main Image",
    #                           command=lambda: update_main_image(trigger_user_function=True))
    #trigger_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Main Frame for Canvases
    main_frame = tk.Frame(root)
    main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Left Canvas (Main Image)
    main_canvas = tk.Canvas(main_frame, width=1024, height=1024, bg="white")
    main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    main_canvas.bind("<Button-1>", on_click)
    main_canvas.bind("<Motion>", _on_mouse_motion)

    # Right Canvas (Pointer Image)
    pointer_canvas = tk.Canvas(main_frame, width=SQUARE_CANVAS_SIZE, height=SQUARE_CANVAS_SIZE, bg="white")
    pointer_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    pointer_canvas.bind("<Button-1>", on_pointer_click)
    main_canvas.bind("<Button-2>", toggle_mouse_motion)  # Right-click to toggle mouse motion functionality

    # Set the initial size for canvases
    main_canvas.config(width=1024, height=1024)
    pointer_canvas.config(width=SQUARE_CANVAS_SIZE, height=SQUARE_CANVAS_SIZE)

    # Start the Tkinter main loop
    root.mainloop()


# Run the visualizer
if __name__ == "__main__":
    create_visualizer()
