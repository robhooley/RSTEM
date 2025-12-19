import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import patches, gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle as p
import easygui as g
import cv2 as cv2
import os
from datetime import datetime
from time import sleep
from tqdm import tqdm
from matplotlib.path import Path as matpath
import fnmatch
import matplotlib.colors as mcolors

from expertpi import api

from expertpi.api import DetectorType as DT, RoiMode as RM

from serving_manager.api import TorchserveRestManager
from RSTEM.app_context import get_app
#comment this out depending on where the script is located
#from RSTEM.utilities import create_circular_mask,check_memory,collect_metadata,downsample_diffraction,array2blo #utilities file in RSTEM directory
#from utilities import create_circular_mask,check_memory,collect_metadata,downsample_diffraction,array2blo



def scan_4D_basic(scan_width_px=128, camera_frequency_hz=4500, use_precession=False):
    """
    Parameters
    ----------
    scan_width_px : int
        Scan width/height in pixels (square raster).
    camera_frequency_hz : int or float
        Camera speed in frames per second (up to 72000).
    use_precession : bool
        Enable/disable precession.

    Returns
    -------
    (image_array, metadata) : tuple
        image_array has shape (scan_width_px, scan_width_px, camY, camX)
    """

    """from expertpi.application import app_state
    #unlock the state if acquisition fails
    app_state.AcqLock.unlock()"""

    app = get_app()

    pixel_time = 1.0 / camera_frequency_hz  # compute once

    #metadata = collect_metadata(
    #    acquisition_type="Camera",
    #    scan_width_px=scan_width_px,
    #    use_precession=use_precession,
    #    pixel_time=pixel_time,
    #    scan_rotation=0,
    #    edx_enabled=False
    #)
    # Ensure STEM detectors are retracted and beam is on axis
    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    if bf_in or haadf_in:
        if bf_in:
            app.detectors.stem.insert_bf(False)
        if haadf_in:
            app.detectors.stem.insert_df(False)
        for _ in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)
    app.scanning.set_off_axis(False)
    sleep(0.2)  # stabilisation after deflector change

    reader = app.acquisition.acquire_camera(pixel_time=np.round(pixel_time,8),total_size=scan_width_px,precession_enabled=use_precession)



    print(f"Acquiring {scan_width_px} x {scan_width_px} px dataset at {camera_frequency_hz} fps")
    #Retrieve first row to infer dtype/shape, then pre-allocate array
    first_row = reader.get_lines(1)
    row_block = first_row.camera # shape: (scan_width_px, camY, camX)
    if row_block.ndim != 4 or row_block.shape[1] != scan_width_px:
        raise RuntimeError(f"Unexpected cameraData shape: {row_block.shape}")
    camY, camX = row_block.shape[2], row_block.shape[3] #read camera size from first row of data
    dtype = row_block.dtype  # keep native dtype to avoid copies/conversions
    image_array = np.empty((scan_width_px, scan_width_px, camY, camX), dtype=dtype, order="C") #Pre-allocate target 4D array: (scanY, scanX, camY, camX)
    #Assign the first row (index 0) directly
    image_array[0, :, :, :] = row_block  # vectorized write
    #Retrieve remaining rows
    for i in tqdm(range(1, scan_width_px), desc="Retrieving data from cache", total=scan_width_px - 1, unit="rows"):
        data = reader.get_lines(1)
        row_block = data.camera  # expected shape: (scan_width_px, camY, camX)
        image_array[i, :, :, :] = row_block #assert row_block.shape == (scan_width_px, camY, camX)
    reader.close()
    print("Array ready")
    return image_array#, metadata


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
    poly = matpath(coords) #draws a polygon around the coordinates extracted from the image
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



#checked ok
def import_tiff_series(scan_width=None,load_metadata=False):
    """Loads in a folder of TIFFs and creates a 4D-array for use with other functions"""
    directory = g.diropenbox("Select directory","Select Directory")
    if scan_width is None: #if scan width variable is empty, prompt user to enter it
        num_files = len(fnmatch.filter(os.listdir(directory), '*.tiff')) #counts how many .tiff files are in the directory
        guessed_scan_width = int(np.sqrt(num_files)) #assumes it is a square acquisition
        scan_width=g.enterbox(f"Enter scan width in pixels, there are {num_files} TIFF files in this folder, "
                                f"scan width might be {guessed_scan_width}","Enter scan width in pixels",
                                default=str(guessed_scan_width))

        scan_width=int(scan_width)
        scan_height = int(num_files/scan_width)

    if load_metadata == False:
        metadata=None
    if load_metadata == True:
        metadata_path = g.fileopenbox("Select metadata file","Select metadata file")
        loading_container = open(metadata_path,"r")
        metadata = json.load(loading_container)

    folder = os.listdir(directory) #formats the directory into proper syntax
    image_list = [] #opens empty list
    for file in tqdm(folder): #iterates through folder with a progress bar
        path = directory+"\\"+file #picks individual images
        if file.endswith(".tiff"):
            image = cv2.imread(path,-1) #loads them with openCV
            image_list.append(image) #adds them to a list of all images

    array = np.asarray(image_list) #converts the list to an array
    cam_pixels_x,cam_pixels_y = image_list[0].shape
    reshaped_array = np.reshape(array,(scan_width,scan_height,cam_pixels_x,cam_pixels_y)) #reshapes the array to 4D dataset shape #TODO confirm ordering of scan width and height with non-square dataset
    if load_metadata == False:
        return reshaped_array #just a data array reshaped to the 4D STEM acquisition shape
    else:
        return (reshaped_array,metadata) #tuple containing the data array and the metadata dictionary
