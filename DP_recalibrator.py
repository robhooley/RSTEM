import piexif
from tkinter import filedialog
import json
from tkinter import Label, StringVar, OptionMenu
import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk  # Tkinter still needs Pillow for GUI, but image processing uses cv2
import numpy as np
import math

import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk


def select_center_and_total_diffraction_angle(filepath):
    """
    Opens a window to let the user click on the image to set the center position,
    enter a total diffraction angle, and confirm or reset the selection.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    dict: Dictionary with center coordinates and total diffraction angle if confirmed.
    """
    # Initialize variables to store center position and total diffraction angle
    center_position = {"x": None, "y": None}
    total_diffraction_angle = None

    # Function to handle mouse click to set the center position
    def on_click(event):
        center_position["x"] = event.x
        center_position["y"] = event.y
        update_display()

    # Function to reset the center position
    def reset_center():
        center_position["x"], center_position["y"] = None, None
        update_display()

    # Function to confirm the selection and close the window
    def confirm_selection():
        nonlocal total_diffraction_angle
        try:
            total_diffraction_angle = float(angle_entry.get())
            if center_position["x"] is None or center_position["y"] is None:
                messagebox.showwarning("Warning", "Please set the center position by clicking on the image.")
                return
            window.quit()  # Close the window
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the total diffraction angle.")

    # Function to update the display with the center marker
    def update_display():
        # Copy the normalized RGB image for display updates
        display_image = normalised_image_rgb.copy()
        if center_position["x"] is not None and center_position["y"] is not None:
            cv2.drawMarker(display_image, (center_position["x"], center_position["y"]),
                           (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        # Update the canvas with the new image
        display_pil = ImageTk.PhotoImage(image=Image.fromarray(display_image))
        canvas.itemconfig(image_on_canvas, image=display_pil)
        canvas.display_image = display_pil  # Prevent garbage collection of the image

    # Load the image
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not load image from path: {filepath}")
        return None

    # Normalize the image to 8-bit and convert to RGB format
    normalised_image = normalise_to_8bit(image)
    normalised_image_rgb = cv2.cvtColor(normalised_image, cv2.COLOR_GRAY2RGB)

    # Set up the Tkinter window
    window = tk.Tk()
    window.title("Set Center Position and Total Diffraction Angle")

    # Convert the normalized image to a format Tkinter can display
    display_pil = ImageTk.PhotoImage(image=Image.fromarray(normalised_image_rgb))

    # Create a canvas to display the image and set the click event
    canvas = tk.Canvas(window, width=normalised_image_rgb.shape[1], height=normalised_image_rgb.shape[0])
    canvas.pack()
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=display_pil)
    canvas.bind("<Button-1>", on_click)

    # Create entry for total diffraction angle
    tk.Label(window, text="Total Diffraction Angle:").pack(pady=5)
    angle_entry = tk.Entry(window)
    angle_entry.pack(pady=5)

    # Add reset and confirm buttons
    button_frame = tk.Frame(window)
    button_frame.pack(pady=10)

    reset_button = tk.Button(button_frame, text="Reset Center", command=reset_center)
    reset_button.grid(row=0, column=0, padx=5)

    confirm_button = tk.Button(button_frame, text="Confirm", command=confirm_selection)
    confirm_button.grid(row=0, column=1, padx=5)

    # Start the Tkinter main loop
    window.mainloop()

    # Return center position and total diffraction angle if confirmed
    return {
        "center_x": center_position["x"],
        "center_y": center_position["y"],
        "total_diffraction_angle": total_diffraction_angle
    }



def get_image_calibrations(filepath):

    #put im try except catch for Expi data, which has no origin or pixel size


    img = cv2.imread(filepath,-1)
    image_shape = img.shape
    calibrations = {}
    calibrations["image shape"] = image_shape
    try:
        metadata_object = piexif.load(filepath)

        pixel_size_mrad = (metadata_object["0th"][piexif.ImageIFD.XResolution][1]/  \
                       metadata_object["0th"][piexif.ImageIFD.XResolution][0])

        description = metadata_object["0th"][270]
        description = description.decode()

        description_split = description.split("\n")

        if description_split[0] == "ImageJ":
            del description_split[0]
        if description_split[-1] == "":
            del description_split[-1]

        for i in description_split:
            value = i.split("=")
            if value[0] == "unit":
                del value
            else:
                calibrations[value[0]]=float(value[1])


        total_diffraction_angle = pixel_size_mrad*image_shape[0]
    except:
        print("Image has no exif tags, user must assist")

        result = select_center_and_total_diffraction_angle(filepath)

        calibrations["xorigin"] = result["center_x"]
        calibrations["yorigin"] = result["center_y"]
        total_diffraction_angle = result["total_diffraction_angle"]

        pixel_size_mrad = total_diffraction_angle/image_shape[0]


    diffraction_semiangle = total_diffraction_angle/2
    wavelength = 3.7014 # pm
    total_size_inv_nm = 4*np.sin(diffraction_semiangle/2/1000)/(wavelength/1e3) #TODO confirm this with Expi data #TOdo not sure why the 4 is needed probably a maths fuckup with total angle and semiangle/2
    total_size_inv_angstrom = total_size_inv_nm/10
    pixel_size_inv_nm = total_size_inv_nm/image_shape[0]
    pixel_size_inv_angstrom = total_size_inv_angstrom / image_shape[0]


    calibrations["total size inverse nanometers"] = total_size_inv_nm
    calibrations["total size mrad"] = total_diffraction_angle
    calibrations["total size inverse angstroms"] = total_size_inv_angstrom
    calibrations["pixel size mrad"]=pixel_size_mrad
    calibrations["pixel size inverse nanometers"]=pixel_size_inv_nm
    calibrations["pixel size inverse angstroms"] = pixel_size_inv_angstrom

    return calibrations,img

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# Function to update the image in the Tkinter window
def update_image_display(image_center_x,image_center_y):
    global img_displayed
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter display
    img_pil = Image.fromarray(img_rgb)  # Convert to PIL format
    img_displayed = ImageTk.PhotoImage(img_pil)  # Convert to ImageTk format
    cv2.drawMarker(img, (int(image_center_x), int(image_center_y)), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20,
                   thickness=1)
    label.config(image=img_displayed)


def convert_distance(distance_px, unit, calibrations):
    if unit == "mrad":
        # Multiply the pixel distance by the corresponding scaling factor
        distance_converted = distance_px * calibrations["pixel size mrad"]
    elif unit == "nm-1":
        distance_converted = distance_px * calibrations["pixel size inverse nanometers"]
    elif unit == "A-1":
        distance_converted = distance_px * calibrations["pixel size inverse angstroms"]
    elif unit == "Pixels":
        distance_converted = distance_px
    return distance_converted, unit  # Return the converted distance and the unit


# Function to handle mouse clicks
def on_click(event):
    global click_count, img, image_center_x, image_center_y, img_displayed, distances

    # Increment click count
    click_count += 1

    # Get the click coordinates
    click_x, click_y = event.x, event.y

    # Calculate distance from the center of the image
    distance_px = calculate_distance(click_x, click_y, image_center_x, image_center_y)
    distance_converted, unit_label = convert_distance(distance_px, selected_unit.get(), calibrations)  # Pass calibrations

    # Append the tuple to distances list: (click count, converted distance, unit)
    distances.append((distance_px,click_count, distance_converted, unit_label))

    # Draw a red cross and number at the clicked position
    cv2.drawMarker(img, (click_x, click_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.putText(img, f"{click_count}", (click_x + 10, click_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"{np.round(distance_converted,2)} {unit_label}", (click_x + 10, click_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    #cv2.drawMarker(img, (image_center_x, image_center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20,
    #               thickness=1)
    # Update the image displayed in Tkinter
    update_image_display(image_center_x, image_center_y)

# Function to save annotated image and distance list to files
def save_results():
    global img, distances, calibrations
    #TODO implement save to directory from popup
    #Dictionary to hold the data for each spot
    data = {}

    # Loop through the distances list and populate the JSON structure
    for distance_px,click_num, _, _ in distances:
        # Convert distances to all the required units
        distance_mrad = distance_px * calibrations["pixel size mrad"]
        distance_inv_nm = distance_px * calibrations["pixel size inverse nanometers"]
        distance_inv_angstrom = distance_px * calibrations["pixel size inverse angstroms"]

        # Add the data for each spot
        data[f"spot_{click_num}"] = {
            "distance_px": np.round(distance_px,2),
            "distance_mrad": np.round(distance_mrad,3),
            "distance_inverse_nanometers": np.round(distance_inv_nm,6),
            "distance_inverse_angstroms": np.round(distance_inv_angstrom,6),
            "distance_angstroms": np.round(1/distance_inv_angstrom,2)
        }

    data["Image calibrations"] = calibrations

    # Save the data as a JSON file
    with open("../acceptance_interface/distances.json", 'w') as f:#TODO set better file location functionality
        json.dump(data, f, indent=4)  # Save with indentation for readability

    print("Distances saved as distances.json")

    # Save the annotated image as well
    annotated_image_path = "../acceptance_interface/annotated_image.jpg" #TODO set better file location functionality
    cv2.imwrite(annotated_image_path, img)
    print(f"Annotated image saved as {annotated_image_path}")

def normalise_to_8bit(image, saturation_percent=1):
    """
       Normalises a 16-bit or 32-bit TIFF image to 8-bit with a specified percent saturation of pixel intensities.

       Parameters:
       image (numpy array): The input image (16-bit or 32-bit).
       saturation_percent (float): Percentage of pixels to saturate (default is 2%).

       Returns:
       normalised_image (numpy array): The normalised 8-bit image with saturation applied.
       """
    # Calculate the lower and upper percentiles for saturation
    lower_percentile = np.percentile(image, saturation_percent)
    upper_percentile = np.percentile(image, 100 - saturation_percent)

    # Ensure a valid range to avoid division by zero
    if upper_percentile == lower_percentile:
        print("Warning: Upper and lower percentiles are equal. Normalization might not work as expected.")
        normalised_image = np.zeros_like(image, dtype=np.uint8)  # Set to black
    else:
        # Clip the image values to these percentiles
        clipped_image = np.clip(image, lower_percentile, upper_percentile)

        # Normalise the image to the 0-255 range after clipping
        normalised_image = ((clipped_image - lower_percentile) / (upper_percentile - lower_percentile) * 255).astype(
            np.uint8)

    colour_normalised = cv2.cvtColor(normalised_image, cv2.COLOR_GRAY2BGR)

    return normalised_image

filepath = filedialog.askopenfilename()
calibrations,img = get_image_calibrations(filepath)


# Initialize the main window
root = tk.Tk()
root.title("TENSOR diffraction measurement")

# Load the image using OpenCV
image_path = filepath
img = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
img= normalise_to_8bit(img,saturation_percent=2)
img_height, img_width = img.shape[:2]

image_center_x, image_center_y = calibrations["xorigin"] , calibrations["yorigin"]#img_width // 2, img_height // 2


# Create a label to display the image
label = Label(root)
label.pack()

# Initialize the click count and distances list
click_count = 0
distances = []  # List to store measured distances

# Create a dropdown for unit selection
selected_unit = StringVar(root)
selected_unit.set("nm-1")  # Default unit is pixels
units = ["A-1", "mrad","Pixels","nm-1"]  # Available units

dropdown = OptionMenu(root, selected_unit, *units)
dropdown.pack()

# Bind the left mouse click event to the function
label.bind("<Button-1>", on_click)

# Display the initial image in the Tkinter window
update_image_display(image_center_x,image_center_y)

# Create a button to save the results (annotated image + distances)
save_button = tk.Button(root, text="Save Results", command=save_results)
save_button.pack()

# Start the Tkinter event loop
root.mainloop()


