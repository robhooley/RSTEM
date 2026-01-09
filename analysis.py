import fnmatch
import json
import os
import pickle as p
from datetime import datetime
from time import time
import cv2
import easygui as g
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from matplotlib.path import Path as MatPath
from tqdm import tqdm

from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT
from expertpi.config import Config

from importlib.util import find_spec

if find_spec("RSTEM.app_context") is not None:
    from RSTEM.utilities import (
        model_has_workers,
        downsample_diffraction,
        create_circular_mask,
    )
else:
    from utilities import (
        model_has_workers,
        downsample_diffraction,
        create_circular_mask,
    )


#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default




def get_spot_positions(image,threshold=0,host=None,model_name="spot_segmentation",logging=True):
    """
    Run a ML spot segmentation model on a diffraction pattern,
    extract detected spot positions and approximate radii, and optionally visualise the
    results.

    This function sends the input image to a TorchServe inference server hosting a spot
    segmentation / detection model. For each detected object, it filters by mean intensity,
    extracts the spot centre coordinates and area, converts the area into an effective
    circular radius normalised to the image size, and overlays the detected spots on the
    input image for quick visual inspection.

    If the specified model is not currently loaded with active workers, the function will
    attempt to scale the model to a single worker before running inference.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array (e.g. diffraction pattern) to be processed by the spot segmentation
        model. The array is passed directly to the inference server and is assumed to be
        compatible with the model's expected input format.
    threshold : float, optional
        Minimum mean intensity required for a detected object to be accepted as a valid
        spot. Objects with ``mean_intensity <= threshold`` are discarded. Default is 0.
    host : str or None, optional
        Hostname or IP address of the TorchServe server. If ``None``, the default inference
        host defined in ``config.inference.host`` is used.
    model_name : str, optional
        Name of the TorchServe model to use for spot segmentation. The model must be
        registered and accessible on the specified server. Default is ``"spot_segmentation",
        also possible is"diffraction_spot_segmentation"``.
    logging : bool, optional
        If ``True``, prints the inference runtime and spot list length to stdout. Default is ``True``.

    Returns
    -------
    spot_list : list of array-like
        List of spot centre coordinates as returned by the model (typically ``[x, y]`` in
        normalised image coordinates).
    spot_radii : list of float
        List of effective spot radii, computed from the segmented area assuming circular
        geometry and normalised by the image height.

    Notes
    -----
    - The effective radius is computed as ``sqrt(area / pi)`` and then normalised by the
      image height (``image.shape[0]``), making the returned radii scale-invariant with
      respect to image size.
    - A matplotlib figure is created showing the input image with detected spot centres
      (red crosses) and corresponding circular outlines (yellow). The figure is displayed
      non-blocking.
    - If the TorchServe server cannot be reached, the function prints an error message and
      returns two empty lists.

    Raises
    ------
    None explicitly. Connection errors to the inference server are caught internally.

    """

    if host == None:
        host = config.inference.host

    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff')  # start server manager
        if not model_has_workers(model_name,host=host):
            manager.scale(model_name=model_name) #scale the server to have 1 process

        if logging:
            process_pre = time()
        results = manager.infer(image=image, model_name=model_name)  # send image to spot detection
        if logging:
            inference_time = time()-process_pre
        spots = results["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the model is available")
        return [],[]
    spot_list = [] #list for all spots
    areas = []
    spot_radii = []
    fig,ax = plt.subplots(1,1)
    shape = image.shape
    for spot in spots:
        if spot["mean_intensity"] > threshold:
            spot_coords = spot["center"]
            area = spot["area"]

            spot_list.append(spot_coords)
            areas.append(area)
            radius = np.sqrt(area / np.pi) / shape[0]
            spot_radii.append(radius)

    ax.imshow(image, vmax=np.average(image * 10), extent=(0, 1, 1, 0), cmap="gray")

    if logging:
        print(f"{len(spot_list)} spots above the threshold intensity were detected in {inference_time:3f}s")

    for coords, radius in zip(spot_list, spot_radii):
        ax.plot(coords[0], coords[1], "r+")
        spot_marker = Circle(xy=coords, radius=radius, color="yellow", fill=False)
        ax.add_patch(spot_marker)


    plt.show(block=False)

    return (spot_list,spot_radii)


def align_image_series(image_series, plot=False, host=None, model_name="TEMRegistration"):
    """
    Align a sequence of 2D images to the first frame using a TorchServe-hosted
    image-registration model and accumulate the aligned result.

    Each image in the input series is registered to the first image via a learned registration model
    (e.g. TEMRegistration) served throughTorchServe. The model is expected to return a normalised
     translation vector, which is converted into pixel shifts and applied using an affine warp.
     All aligned frames are summed to produce an accumulated image, and the per-frame shifts are recorded.

    Optionally,plots can be generated to visualise the reference image, final
    frame, summed image, and measured drift trajectory.

    Parameters
    ----------
    image_series : list of numpy.ndarray
        Non-empty list of 2D images to be aligned. All images must have identical shape and
        represent the same field of view. The first image in the list is used as the fixed
        reference for all registrations.
    plot : bool, optional
        If ``True``, displays a multi-panel matplotlib figure showing the reference image,
        the final frame in the series, the summed aligned image, and a scatter plot of the
        measured x/y shifts. Default is ``False``.
    host : str or None, optional
        Hostname or IP address of the TorchServe inference server. If ``None``, the value is
        taken from ``config.inference.host``.
    model : str, optional
        Name of the registration model deployed.  Default is ``"TEMRegistration"``. also possible are:
        "TEMRoma", "TEMRomaTiny"

    Returns
    -------
    translated_list : list of numpy.ndarray
        List of aligned images, each warped into the reference frame and returned as
        ``uint16`` arrays.
    summed_image : numpy.ndarray
        Floating-point image formed by summing all aligned frames. No normalisation or
        averaging is applied.
    shifts : list of tuple of float
        List of measured translations for each frame in pixel units, given as
        ``(x_shift_px, y_shift_px)`` relative to the reference image.

    Raises
    ------
    Exception
        If ``image_series`` is empty or not a list, or if the registration model cannot be
        contacted on the specified host.
    RuntimeError
        If the registration model returns an invalid or malformed result (e.g. missing or
        non-finite translation values).

    """

    if host is None:
        host = config.inference.host  #pull from config

    if not isinstance(image_series, list) or len(image_series) == 0:
        raise Exception("Dataset must be a non-empty list of images")

    initial_image = np.array(image_series[0], dtype=np.float64)
    H, W = initial_image.shape
    shifts = []
    translated_list = []

    manager = TorchserveRestManager(
        inference_port='8080',
        management_port='8081',
        host=host,
        image_encoder='.tiff'
    )
    try:
        manager.scale(model_name=model_name)
    except Exception:
        raise Exception(f"{model_name} cannot be contacted")

    acc = np.zeros((H, W), dtype=np.float64)

    for idx in tqdm(range(len(image_series)), desc="Aligning frames", unit="img"):
        moving_u16 = np.array(image_series[idx])
        moving_f = moving_u16.astype(np.float32,)

        reg_input = np.concatenate([initial_image, moving_f], axis=1)
        registration_values = registration_model(
            reg_input,
            model_name,
            host=host,
            port="8080",
            image_encoder='.tiff'
        )

        if not isinstance(registration_values, (list, tuple)) or len(registration_values) == 0:
            raise RuntimeError("registration_model returned no results")
        rv0 = registration_values[0]
        if not isinstance(rv0, dict) or "translation" not in rv0:
            raise RuntimeError("registration_model result missing 'translation'")
        translation = rv0["translation"]
        if (not isinstance(translation, (list, tuple)) or len(translation) != 2 or
                not np.all(np.isfinite(translation))):
            raise RuntimeError(f"Invalid 'translation' payload: {translation!r}")

        dx_norm, dy_norm = translation

        x_pixels_shift = dx_norm * W
        y_pixels_shift = dy_norm * H
        shifts.append((x_pixels_shift, y_pixels_shift))

        M = np.float32([[1, 0, x_pixels_shift], [0, 1, y_pixels_shift]])

        warped_f = cv2.warpAffine(
            moving_f,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        acc += warped_f.astype(np.float64)

        warped_u16 = np.clip(warped_f, 0, 65535).astype(np.uint16)
        translated_list.append(warped_u16)

    summed_image = acc

    if plot:
        xs, ys = zip(*shifts) if shifts else ([], [])
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)
        fig.set_size_inches(15, 5)
        ax1.imshow(initial_image, cmap="gray")
        ax1.title.set_text("Initial Image")
        plt.setp(ax1, xticks=[], yticks=[])
        ax2.imshow(image_series[-1], cmap="gray")
        ax2.title.set_text(f"Final Image")
        plt.setp(ax2, xticks=[], yticks=[])
        ax3.imshow(summed_image, cmap="gray")
        ax3.title.set_text(f"Summation of {len(image_series)} images")
        plt.setp(ax3, xticks=[], yticks=[])
        ax4.scatter(xs, ys, s=10)
        xs, ys = zip(*shifts) if shifts else ([], [])
        ax4.set_title("Measured drift")
        ax4.set_xlabel("x shift [px]")
        ax4.set_ylabel("y shift [px]")
        if xs and ys:
            t = np.arange(len(xs))  # 0..N-1 (acquisition index)
            # RdYlBu maps low->red, high->blue (so start=red, end=blue)
            sc = ax4.scatter(xs, ys, c=t, cmap="RdYlBu", s=16)
            # (optional) faint trajectory line for context
            ax4.plot(xs, ys, linewidth=0.8, alpha=0.4)
            # 1:1 data scale and square axes panel
            ax4.set_aspect('equal', adjustable='datalim')
            ax4.set_box_aspect(1)
            ax4.set_anchor('C')
            ax4.grid(True, alpha=0.2)
            # Colorbar keyed to acquisition index
            cb = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
            cb.set_label("Frame index (0 = start)")
        ax4.set_anchor('C')  # center the square axis in its slot
        ax4.autoscale_view()  # respect current data limits
        plt.show()

    return translated_list, summed_image, shifts

def find_ronchigram_center(image,host=None,model_name="RonchigramCenter",logging=True):
    """
    Infer the optical centre of a Ronchigram using a TorchServe-hosted segmentation model
    and select the most probable centre based on segmented area.

    This function sends a Ronchigram image to a TorchServe inference server running a
    dedicated centre-detection / segmentation model. The model is expected to return one
    or more segmented objects corresponding to candidate Ronchigram regions. Each
    candidate provides a centre coordinate and an associated area. The function selects
    the centre associated with the largest segmented area, which is assumed to correspond
    to the true Ronchigram centre.

    A diagnostic plot is generated showing the input image with the selected centre
    overlaid for visual verification.

    Parameters
    ----------
    image : numpy.ndarray
        2D Ronchigram image to be analysed. The array is passed directly to the TorchServe
        model and must be compatible with the model’s expected input format.
    host : str or None, optional
        Hostname or IP address of the TorchServe inference server. If ``None``, the default
        value defined in ``config.inference.host`` is used.
    model_name : str, optional
        Name of the TorchServe model used for Ronchigram centre detection. Default is
        ``"RonchigramCenter"``.
    logging : bool, optional
        If ``True``, prints the inference runtime to stdout. Default is ``True``.

    Returns
    -------
    None
        This function does not currently return the detected centre. The selected centre
        (corresponding to the largest segmented area) is visualised on the input image.
        If the model cannot be reached, the function returns two empty lists.

    Raises
    ------
    None explicitly. Connection errors to the inference server are caught internally.

    Notes
    -----
    - The function assumes that the correct Ronchigram centre corresponds to the segmented
      object with the largest area.
    - The model output is expected to follow the TorchServe segmentation convention:
      ``results["segmentation"][0]["objects"]``, where each object contains ``"center"``
      and ``"area"`` fields.
    - The displayed image uses normalised coordinates (extent 0–1) for plotting the
      detected centre.
    - The matplotlib figure is shown in non-blocking mode to allow continued execution
      in interactive workflows.

    """

    if host == None:
        host = config.inference.host

    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff')  # start server manager
        if not model_has_workers(model_name,host=host):
            manager.scale(model_name=model_name) #scale the server to have 1 process

        if logging:
            process_pre = time()
        results = manager.infer(image=image, model_name=model_name)  # send image to spot detection
        if logging:
            inference_time = time()-process_pre
            print(f"Inference Time: {inference_time:3f}s")
        fits = results["segmentation"][0]["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the model is available")
        return [],[]
    fit_list = [] #list for all spots
    areas = []

    fig,ax = plt.subplots(1,1)
    shape = image.shape
    for fit in fits:
        center = fit["center"]
        area = fit["area"]
        fit_list.append(center)
        areas.append(area)

    max_area_index = areas.index(max(areas))
    center_with_max_area = fit_list[max_area_index]

    ax.imshow(image, vmax=np.average(image * 10), extent=(0, 1, 1, 0), cmap="gray")
    ax.plot(center_with_max_area[0], center_with_max_area[1], "r+")
    plt.show(block=False)

def selected_area_diffraction(data_array):
    """Takes a 4D data array as produced by scan_4D_basic and allows the user to select virtual apertures in the image
    to integrate diffraction from"""

    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]
    else:
        image_array = data_array

    camera_data_shape = image_array[0][0].shape #shape of first image to get image dimensions
    dataset_shape = image_array.shape[0],image_array.shape[1] #scanned region shape
    radius = 30  # pixels for rough VBF image construction
    VBF_intensity_list = [] #empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_radius=radius)
    for row in image_array: #iterates through array rows
        for pixel in row: #in each row iterates through pixels
            VBF_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
            VBF_intensity_list.append(VBF_intensity) #adds to the list

    VBF_intensity_array = np.asarray(VBF_intensity_list) #converts list to array
    VBF_intensity_array = np.reshape(VBF_intensity_array, (dataset_shape[0], dataset_shape[1])) #reshapes array to match image dimensions
    plt.figure(figsize=(10,10)) #sets figure size large enough for clear display
    plt.imshow(VBF_intensity_array) #plots VBF

    plt.title("Click to add points, right click to remove previous point, middle click to finsh and complete polygon")
    plt.gray() #sets grayscale
    coords = plt.ginput(n=-1,show_clicks=True,timeout=0) #use user mouse input to define the integration region
    plt.close()
    poly = MatPath(coords) #draws a polygon around the coordinates extracted from the image
    all_pixel_coordinates = [] #list for all possible coordinates
    inside_pixels = [] #list for coordinates within the polygon
    """this is hacky but works"""
    points = image_array.shape[0],image_array.shape[1]
    for i in range(0,points[0]):
        for j in range(0,points[1]):
            all_pixel_coordinates.append([i,j])  # this can probably be done more elegantly
    for pixel in all_pixel_coordinates: #for all possible pixels
        is_inside = poly.contains_point(pixel) #checks if the pixel is inside the polygon
        if is_inside == True: #if so, then
            inside_pixels.append(pixel) #adds internal pixel to list

    number_of_summed_patterns = len(inside_pixels) #number of pixels within the polygon
    subset_DP_list = [] #empty list to take the diffraction patterns inside the polygon
    for pixel in inside_pixels: #for each pixel inside the polygon
        pattern = image_array[pixel[1]][pixel[0]] #take the diffraction pattern
        pattern.astype(np.float64) #convert it to 64 bit (better for summation)
        subset_DP_list.append(pattern) #add it to the empty list
    subset_array = np.asarray(subset_DP_list)
    subset_summed_DP = np.sum(subset_array,0,dtype=np.float64)
    zero_excluded_max = max(subset_summed_DP[~integration_mask]) #highest intensity outside the zero order disk

    export_polygon = patches.Polygon(coords,fill=False,edgecolor="red") #creates a polygon from the user defined integration region
    export_fig = plt.figure(figsize=(10, 10)) #defines an empty figure to add the export to
    export_axis = export_fig.add_subplot() #add plot to exporting figure
    plt.setp(export_axis, xticks=[], yticks=[])  # removes tick markers
    export_axis.add_patch(export_polygon) #adds a circle for every mask used

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0) #removes whitespace
    plt.margins(0, 0) #removes white space
    plt.imshow(VBF_intensity_array) #add image to canvas

    canvas = FigureCanvasAgg(export_fig) #defines a canvas
    export_fig.canvas.draw() #plots the canvas
    buffered = canvas.buffer_rgba() #writes it to a buffer
    annotated_image = np.asarray(buffered) #converts the buffer to an image
    plt.close() #close the figure

    polygon = patches.Polygon(coords, fill=False, edgecolor="red")
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10)) #defines a 1x2 plot of reasonable size
    ax1.title.set_text("Integration region")

    ax1.add_patch(polygon)  # shows the polygon of the integration window in the calibrated image
    ax1.imshow(VBF_intensity_array)
    ax2.imshow(subset_summed_DP.astype(np.float64),vmin=0,vmax=zero_excluded_max) #show the image and scale to that intensity
    ax2.title.set_text(f"Summed diffraction pattern from {number_of_summed_patterns} patterns") #add title
    plt.show() #show plot

    return subset_summed_DP, annotated_image #return the summed image, the polygon and the VBF image

#checked ok
def multi_VDF(data_array,radius=None):
    """
     Produces virtual dark field images from user selected points
     Arguments:
         image_list : numpy array of all diffraction patterns
         radius : virtual aperture radius in pixels, default is 10 pixels
    Returns:
        Summed diffraction pattern taken from all pixels
        List of Virtual Dark Field images
     """

    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0]

    else:
        image_array = data_array

    if radius is None:
        radius = 20

    dataset_shape = image_array.shape[0], image_array.shape[1]
    dp_shape = image_array[0][0].shape

    subset_images = []
    for i in range(0,5*dataset_shape[0]): #take a number of random images from the dataset
        random_image = image_array[np.random.randint(0,dataset_shape[0])][np.random.randint(0,dataset_shape[1])]
        subset_images.append(random_image.astype(np.float64))

    sum_diffraction = sum(subset_images) #sums the random images to make a representative diffraction pattern
    av_int = np.average(sum_diffraction) #calculates the average intensity of this pattern
    plt.figure(figsize=(10,10)) #defines plot size
    plt.gray() #sets it to grayscale
    plt.title("Click to place virtual apertures, right click to remove and middle click to finish, max masks 8")
    plt.imshow(sum_diffraction,vmax=av_int*10) #plots the representative diffraction and scales the intensity
    mask_list = plt.ginput(n=8,show_clicks=True,timeout=0) #user interacts to define mask positions
    plt.close() #closes the plot
    all_mask_intensities = [] #empty list for the mask data

    integration_masks = []
    for mask_coords in mask_list:
        integration_mask = create_circular_mask(dp_shape[0], dp_shape[1], mask_center_coordinates=mask_coords,
                                                mask_radius=radius)
        integration_masks.append(integration_mask)

    for row in tqdm(image_array):
        for pixel in row:
            mask_intensities = [] #empty list of mask intensities per pixel
            for mask_coords in range(len(mask_list)): #for each mask
                integration_mask = integration_masks[mask_coords] #takes mask from list and applies it
                mask_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
                mask_intensities.append(mask_intensity) #adds to the list

            all_mask_intensities.append(mask_intensities) #adds each pixels list to a list (nested lists)

    DF_images = [] #empty list for DF images to be added to
    for mask in range(len(mask_list)): #for every mask in the list
        DF_output = [i[mask] for i in all_mask_intensities] #this works but I don't really understand why
        DF_output = np.reshape(DF_output,(dataset_shape)) #reshapes the DF intensities to the scan dimensions
        DF_images.append(DF_output) #adds DF images to a list

    if len(mask_list) ==1:
        grid_rows,grid_cols=1,2 #1x2 plot for 1 mask
    elif len(mask_list) ==2:
        grid_rows, grid_cols = 1, 3 #1x3 plot for 2 masks +1DP
    elif len(mask_list) == 3:
        grid_rows, grid_cols = 2, 2 #2x2 plot for 3 masks +1DP
    elif len(mask_list) <= 5:
        grid_rows, grid_cols = 2, 3
    elif len(mask_list) <= 7:
        grid_rows, grid_cols = 2,4  #2x4 plot for 7 masks +1DP
    elif len(mask_list) ==8:
        grid_rows, grid_cols = 3, 3 #3x3 plot for 8 +1DP

    plot_grid = gridspec.GridSpec(grid_rows, grid_cols) #defines the plot grid

    av_int = np.average(sum_diffraction)  # calculates the average intensity in the diffraction pattern, for display scaling
    colors = list(mcolors.TABLEAU_COLORS)  # sets a list of colours
    export_fig = plt.figure(figsize=(10, 10))
    export_axis = export_fig.add_subplot()
    plt.imshow(sum_diffraction, vmin=0, vmax=av_int*10)
    plt.setp(export_axis, xticks=[], yticks=[])  # removes tick markers
    for coords in range(len(mask_list)):
        circle = plt.Circle(mask_list[coords], radius=radius, color=colors[coords], fill=True,alpha=0.3) #defines the mask annotation
        export_axis.add_patch(circle) #adds a circle for every mask used

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0) #removes whitespace
    plt.margins(0, 0) #removes white space
    canvas = FigureCanvasAgg(export_fig) #defines a canvas
    export_fig.canvas.draw() #plots the canvas
    buffered = canvas.buffer_rgba() #writes it to a buffer
    annotated_image = np.asarray(buffered) #converts the buffer to an image
    plt.close()

    fig = plt.figure(figsize=(grid_cols*5,grid_rows*5)) #makes a figure of reasonable

    ax=fig.add_subplot(plot_grid[0]) #adds the ax axis to the plot grid in the zero position
    ax.title.set_text("Summed diffraction with integration windows") #adds title
    av_int = np.average(sum_diffraction) #calculates the average intensity in the diffraction pattern, for display scaling
    plt.imshow(annotated_image)
    plt.setp(ax, xticks=[], yticks=[]) #removes tick markers
    colors = list(mcolors.TABLEAU_COLORS) #sets a list of colours

    for i in range(len(mask_list)): #for every mask used
        ax=fig.add_subplot(plot_grid[i+1]) #selects the plotting grid position for the DF to go in
        ax.imshow(DF_images[i],cmap="gray") #adds the DF image
        ax.title.set_text(i+1) #mask number
        ax.title.set_color(colors[i]) #colour code for title
        ax.spines['bottom'].set_color(colors[i])#colour rim
        ax.spines['bottom'].set_linewidth(3)#line thickness
        ax.spines['top'].set_color(colors[i])#colour rim
        ax.spines['top'].set_linewidth(3)#line thickness
        ax.spines['right'].set_color(colors[i])#colour rim
        ax.spines['right'].set_linewidth(3)#line thickness
        ax.spines['left'].set_color(colors[i]) #colour rim
        ax.spines['left'].set_linewidth(3) #line thickness
        plt.setp(ax, xticks=[], yticks=[]) #removes ticks
    plt.gray() #sets plots to be grayscale
    plt.show() #shows the plot
    #plt.close()

    return annotated_image ,sum_diffraction,DF_images #annotated image is scaled to show the final figure scale which is small #TODO make this better, maybe plot it again before export?

def save_4D_data(data_array,format=None,output_resolution=None,use_datetime_session=True):
    """
    Handles data saving for scan_4D_basic.
    Parameters
    ----------
    data_array : tuple | np.ndarray
        From scan_4D_basic, either (image_array, metadata) or just image_array.
    format : str | None
        One of "TIFFs", "Numpy array", "Pickle", "NanoMEGAS Block".
        If None, a dialog is shown.
    output_resolution : {128, 256, "No Downscaling", None}
        Used to set scaling of diffraction patterns to square only.
        If None, a dialog is shown (for non-NanoMEGAS which has automatic downscaling).
    use_datetime_session : bool
        If True, filenames include a date-time token. Otherwise count files in dir and +1.
    """
    # ---- Unpack and validate input -----------------------------------------
    if isinstance(data_array, (tuple, list)) and len(data_array) >= 2:
        image_array, metadata = data_array[0], data_array[1]
        print("Metadata exists")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict if provided.") #reject if not dictionary
    else:
        image_array, metadata = data_array, None
        print("Metadata not present")

    if not hasattr(image_array, "ndim") or image_array.ndim != 4:
        raise ValueError("Expected a 4D array shaped (scan_y, scan_x, pat_y, pat_x).")

    # ---- Directory selection ------------------------------------------------
    directory = g.diropenbox("Select directory to save to", "Select save directory")
    if not directory:
        print("Save cancelled (no directory).")
        return None
    os.makedirs(directory, exist_ok=True)

    # Base filename
    base = os.path.join(directory, "4D_STEM")
    session_suffix = ""
    if use_datetime_session:
        session_suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    # ---- Format selection ---------------------------------------------------
    valid_formats = ["TIFFs", "Numpy array", "Pickle", "NanoMEGAS Block"]
    if format is None:
        format = g.choicebox("Select format for data to be saved","Select format for data to be saved", valid_formats,preselect=1)
        if format is None:
            print("Save cancelled (no format).")
            return None

    if format not in valid_formats:
        raise ValueError(f"Unsupported format: {format}")

    def _normalize_out_res(val):
        if val is None:
            return None
        if isinstance(val, str):
            if val.strip().lower() in {"no downscaling", "no_downscaling", "none"}: #handles nonstandard strings
                return None
            if val.strip().isdigit():
                return int(val.strip()) #gui box returns strings, stripping to int
            # Let this raise below
        return int(val)

    if output_resolution is None and format != "NanoMEGAS Block":
        choice = g.choicebox("Enter pattern output resolution","Output resolution",choices=["128", "256", "No Downscaling"],preselect=2)
        if choice is None:
            print("Save cancelled")
            return None
        output_resolution = _normalize_out_res(choice)
    else:
        # Normalize programmatic input too
        if format != "NanoMEGAS Block":
            output_resolution = _normalize_out_res(output_resolution)
        else:
            output_resolution = None  # Always skip for NanoMEGAS

    # ---- Optional downsampling ---------------------------------------------
    if output_resolution is not None:
        print(f"Resizing diffraction patterns to {output_resolution}x{output_resolution}")
        # expects: downsample_diffraction(array, size, mode)
        image_array = downsample_diffraction(image_array, int(output_resolution), "sum")
        if metadata is not None:
            metadata = dict(metadata)  # avoid mutating caller's dict
            metadata["Diffraction rescaled to"] = int(output_resolution)
    # Frame count (from current image_array)
    scan_y, scan_x, _, _ = image_array.shape
    n_frames = scan_y * scan_x
    metadata_path = None
    # ---- Save branches
    if format == "TIFFs":
        print(f"Saving {n_frames} files as .tiff")
        i = 0
        for row in tqdm(image_array, desc="Writing TIFFs"):
            for image in row:
                if image.dtype != np.uint16:
                    img16 = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
                else:
                    img16 = image
                if use_datetime_session:
                    tiff_name = f"{base}{session_suffix}_{i:06}.tiff"
                else:
                    tiff_name = f"{base}_{i:06}.tiff"
                cv2.imwrite(tiff_name, img16)
                i += 1
        if metadata is not None:
            if use_datetime_session:
                metadata_path = os.path.join(directory, f"tiff_series_metadata{session_suffix}.json")
            else:
                metadata_path = os.path.join(directory, "tiff_series_metadata.json")
        print("Saving complete")
    elif format == "Numpy array":
        if use_datetime_session:
            npy_path = f"{base}{session_suffix}.npy"
        else:
            # “count files and +1”
            num_files_in_dir = len(fnmatch.filter(os.listdir(directory), '*.npy'))
            npy_path = f"{base}_{num_files_in_dir + 1}.npy"
        print(f"Saving to numpy array, {npy_path}")
        np.save(npy_path, image_array)
        if metadata is not None:
            if use_datetime_session:
                metadata_path = f"{base}{session_suffix}_NPY_metadata.json"
            else:
                metadata_path = os.path.join(
                    directory, f"4D_STEM_{num_files_in_dir + 1}_NPY_metadata.json")
        print("Saving complete")
    elif format == "Pickle":
        if use_datetime_session:
            pdat_path = f"{base}{session_suffix}.pdat"
            metadata_path = f"{base}{session_suffix}_pdat_metadata.json" if metadata is not None else None
        else:
            num_files_in_dir = len(fnmatch.filter(os.listdir(directory), '*.pdat'))
            pdat_path = f"{base}_{num_files_in_dir + 1}.pdat"
            metadata_path = (
                os.path.join(directory, f"4D_STEM_{num_files_in_dir + 1}_pdat_metadata.json")
                if metadata is not None else None)
        if metadata is not None:
            print(f"Saving as Pickle with metadata baked into {os.path.basename(pdat_path)}")
            with open(pdat_path, "wb") as f:
                p.dump((image_array, metadata), f)
        else:
            print(f"Saving as Pickle without metadata baked into {os.path.basename(pdat_path)}")
            with open(pdat_path, "wb") as f:
                p.dump(image_array, f)
        print("Pickling complete")
    #elif format == "NanoMEGAS Block":
    #    if use_datetime_session:
    #        blo_path = f"{base}{session_suffix}.blo"
    #        metadata_path = f"{base}{session_suffix}_blo_metadata.json" if metadata is not None else None
    #    else:
    #        num_files_in_dir = len(fnmatch.filter(os.listdir(directory), '*.blo'))
    #        blo_path = f"{base}_{num_files_in_dir + 1}.blo"
    #        metadata_path = (
    #            os.path.join(directory, f"4D_STEM_{num_files_in_dir + 1}_blo_metadata.json")
    #            if metadata is not None else None)
    #    print(f"Saving as blo file {blo_path}")
    #    array2blo(data=image_array, meta=metadata, filename=blo_path)
    if metadata is not None and metadata_path is not None:
        with open(metadata_path, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)
    print("Export completed")

