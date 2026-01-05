from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import cv2 as cv2
import psutil
from tqdm import tqdm
import easygui as g
import numpy as np
from time import sleep
import scipy.constants
import matplotlib.colors as mcolors
import random
from rsciio.blockfile import file_writer
import os
import tifffile as tiff
from time import time
import matplotlib.pyplot as plt

#from expert_pi.__main__ import window

#from expert_pi.__main__ import window
#from expert_pi import grpc_client
#from expert_pi.client import grpc_client #0.2.2
#from expert_pi.app import scan_helper
#from expert_pi.grpc_client.modules._common import DetectorType as DT
#from expert_pi.grpc_client.enums import DetectorType as DT #0.2.2
#from expert_pi.app import app
#from expert_pi.gui import main_window

from expertpi.api import DetectorType as DT, RoiMode as RM, CondenserFocusType as CFT
from serving_manager.tem_models.specific_model_functions import registration_model
from serving_manager.management.torchserve_rest_manager import TorchserveRestManager
from expertpi.config import Config
from RSTEM.app_context import get_app

#cache client is now binary client
#app.api is grpc_client

#scan helper is gone

#acquisition is now set differently for camera and stem

#acquisition now returns a reader, data = reader.get_frame()
#for STEM the data is cached and can be read with get_frame
#for camera use reader.get_pixels() or get_lines() lines is my current behaviour
#data must be read from the reader, or the reader must be closed to flush the cache and allow next acquisition
#using app.command is synched to UI, but app.api is not (old behaviour)


#config = Config()
config = Config(r"C:\Users\stem\Documents\Rob_coding\ExpertPI-0.5.1\config.yml") # path to config file if changes have been made, otherwise comment out and use default


def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)
    mask = dist_from_center <= mask_radius
    return mask


def spot_radius_in_px(data_array): #TODO refactor to be smarter
    """Takes a data array and works out the diffraction spot radius in pixels from the metadata"""
    if type(data_array) is tuple: #checks for metadata dictionary
        image_array = data_array[0] #splits the tuple to image array and metadata dictionary
        metadata = data_array[1] #splits the tuple to image array and metadata dictionary
        shape_4D = image_array.shape #shape of dataset
        dp_shape = shape_4D[2],shape_4D[3] #number of pixels in DP
        print("Metadata is present, calculating diffraction spot size in pixels")
        convergence_semiangle = metadata.get("Convergence semiangle (mrad)")
        diffraction_angle = metadata.get("Diffraction semiangle (mrad)")
        mrad_per_pixel = (diffraction_angle*2)/dp_shape[0] #angle calibration per pixel
        convergence_pixels = convergence_semiangle/mrad_per_pixel #convergence angle in pixels
        pixel_radius = convergence_pixels #semi-angle and radius

    else:
        print("Metadata is not present, assuming 10 pixels for spot radius")
        pixel_radius = 10

    return pixel_radius

def check_memory(camera_frequency_hz,scan_width_px,roi_mode=False,verbose=False):
    """Checks the current acquisitions size and checks the amount of RAM available to see if it will fit cleanly
    Returns True or False if the dataset will fit in the RAM"""
    if camera_frequency_hz < 2250:
        bit_depth = 16
    else:
        bit_depth=8

    if roi_mode==128:
        cam_size=512*128
    elif roi_mode==256:
        cam_size=512*256
    else:
        cam_size=512*512

    predicted_dataset_size = (scan_width_px*scan_width_px)*cam_size*bit_depth #in bits
    predicted_dataset_size_gbytes = (predicted_dataset_size/8)/1e9
    predicted_dataset_size_with_buffer = predicted_dataset_size_gbytes*1.1
    free_ram = psutil.virtual_memory().free/1e9
    if verbose:
        print(f"There are {free_ram} Gb of RAM available,dataset predicted to be {predicted_dataset_size_with_buffer}Gb")
    if free_ram>predicted_dataset_size_with_buffer:
        will_work = True
    else:
        will_work = False
    return will_work

def create_scalebar(ax,scalebar_size_pixels,metadata):
    print(scalebar_size_pixels,"scalebar requested in pixels")
    pixel_size = metadata["Pixel size (nm)"]*1e6
    scalebar_nm = int(scalebar_size_pixels*pixel_size)
    print("scalebar in nanometers",scalebar_nm)
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                        scalebar_nm,f"{scalebar_nm}nm", 'lower right',
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)
    ax.add_artist(scalebar)

def calculate_wavelength(energy):
    """energy in electronvolts -> return wavelength in picometers"""
    phir = energy*(1 + scipy.constants.e*energy/(2*scipy.constants.m_e*scipy.constants.c**2))
    g = np.sqrt(2*scipy.constants.m_e*scipy.constants.e*phir)
    k = g/scipy.constants.hbar
    wavelength = 2*np.pi/k
    return wavelength*1e12  # to picometers

def generate_colourlist(num_colors_needed, mode=None):
    if mode == "Explore": #this is only used as a joke
        print("Using Explores colour palette (Sorry Raman...)")
        reasonable_colors = ["#FF69B4", '#FF00FF', '#F3CFC6', '#FA8072', "#DA70D6","#FAA0A0","#F89880","#A95C68","#E30B5C","#FF10F0","#D8BFD8","#E37383",
                             "#E0BFB8","#9F2B68","#F2D2BD","#DE3163","#FF7F50"]
        color_list_output = reasonable_colors[:num_colors_needed]
    else:
        colorlist = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        reasonable_colors = ["#FF8C00",'#9932CC','#2E8B57','#FA8072','#A0522D','#87CEEB','#D8BFD8','#FF6347','#6B8E23',"#7CFC00",'#E6E6FA','#FF69B4','#DB7093','#FFD700','#B0C4DE']
        if num_colors_needed > len(reasonable_colors):

            num_extra_needed = num_colors_needed-len(reasonable_colors)
            while num_extra_needed != 0 :
                random_pick = random.choice(list(colorlist.values()))
                if random_pick not in reasonable_colors:
                    reasonable_colors.append(random_pick)
                num_extra_needed = num_colors_needed - len(reasonable_colors)
        color_list_output = reasonable_colors[:num_colors_needed]

    return color_list_output

def generate_colourmaps(num_colors, num_bins=100, mode=None):
    colormaps = []
    color_list = generate_colourlist(num_colors, mode)
    for color in color_list:
        colors_ = [mcolors.to_rgb('black'), mcolors.to_rgb(color)]  #
        cmap_name = 'black_' + color
        colormaps.append(mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_, N=num_bins))

    return colormaps

#TODO refacctored but needs more testing and checking in functions
def collect_metadata(acquisition_type=None,scan_width_px=None,use_precession=False,pixel_time=None,edx_enabled=False,num_frames=None,scan_height=None): #TODO needs testing
    """Extracts acquisition conditions from the microscope and stores them in a dictionary
    Parameters:
    acquisition_type: String "STEM" or" camera"/"4d-stem,
    scan_width_px: number of pixels in x direction,
    use_precession: True or False,
    pixel_time: in seconds,
    scan_rotation: degrees scan rotation applied,
    edx_enabled: True or False,
    camera_pixels: 128, 256, or 512,
    num_frames: for multiframe EDX acquisitions
    """

    """Get all conditions to reduce unnecessary hardware call duplication"""

    app = get_app()

    if scan_width_px is None:
        scan_width_px = app.scanning.get_pixel_count().value

    fov = app.scanning.get_fov() #get the current scanning FOV
    pixel_size_nm = (fov/scan_width_px)*1e9 #pixel size in nanometers
    time_now = datetime.now()
    acquisition_time = time_now.strftime("%d_%m_%Y %H_%M")
    energy= app.api.gun.get_high_voltage()
    wavelength = calculate_wavelength(energy)
    camera_angle =  app.detectors.camera.get_camera_angle()
    pixel_size_inv_angstrom = (2* camera_angle)*1e-3/(
            wavelength*0.01)/512 #assuming 512 pixels
    probe_current = app.optics.get_current()  # amps
    probe_size = app.optics.get_diameter()  # meters

    optical_mode = app.api.microscope.get_optical_mode().name

    max_angles = app.api.projection.get_max_detector_angles()


    if pixel_time == None: #only needed for acquisitions from the console that use the UI dwell time
        pixel_time = app.scanning.get_pixel_time()

    electrons_per_amp = 1 / scipy.constants.elementary_charge
    electrons_in_probe = electrons_per_amp * probe_current  # electrons in probe per second
    electrons_per_pixel_dwell = electrons_in_probe * pixel_time  # divide by dwell time converted to seconds
    """pixel size calculation"""
    pixel_size = fov / scan_width_px  # in meters
    pixel_area = pixel_size ** 2
    electrons_per_meter_square_pixel = electrons_per_pixel_dwell / pixel_area
    """probe size calculation"""

    probe_area = np.pi * (probe_size / 2) ** 2  # assume circular probe
    electrons_per_meter_square_probe = electrons_per_pixel_dwell / probe_area
    """Calculate pixel size to probe size ratio"""
    pixel_to_probe_ratio = pixel_size / probe_size

    dose_rate_probe_angstroms = electrons_per_meter_square_probe * 1e-20 / pixel_time
    dose_rate_pixel_angstroms = electrons_per_meter_square_pixel * 1e-20 / pixel_time

    microscope_info = { #creates a dictionary of microscope parameters
    "Acquisition date and time":acquisition_time,
    "Optical mode":optical_mode,
    "High Tension (kV)":energy/1e3,
    "Probe current (pA)" : float(np.round(probe_current*1e12,2)),
    "Convergence semiangle (mrad)" : float(np.round(app.optics.get_angle()*1e3,2)),
    "Beam diameter (d50) (nm)" : float(np.round(probe_size*1e9,2)),
    "FOV (um)" : fov*1e6,
    "Pixel size (nm)" : float(np.round(pixel_size_nm,2)),
    "Pixel time (s)": pixel_time,
    "Scan rotation (deg)": float(np.rad2deg(app.scanning.get_scanning_rotation())),
    "Pixel dose (e-nm-2)": float(np.round(electrons_per_meter_square_pixel * 1e-18,3)),
    "Probe dose (e-nm-2)": float(np.round(electrons_per_meter_square_probe * 1e-18,3)),
    "Pixel dose (e-A-2)": float(np.round(electrons_per_meter_square_pixel * 1e-20,3)),
    "Probe dose (e-A-2)": float(np.round(electrons_per_meter_square_probe * 1e-20,3)),
    "Pixel dose rate (e-A-2s-1)": float(np.round(dose_rate_pixel_angstroms,3)),
    "Probe dose rate (e-A-2s-1)": float(np.round(dose_rate_probe_angstroms,3)),
    "Ratio of probe size to pixel size": float(np.round(pixel_to_probe_ratio,3))}

    microscope_info["Acquisition type"]= acquisition_type

    microscope_info["Scan width (px)"] = scan_width_px
    if scan_height is not None:
        microscope_info["Scan height (px)"]: scan_height

    if acquisition_type.lower() == "stem":
        HAADF_inserted = app.api.stem_detector.get_is_inserted(DT.HAADF)


        microscope_info["Dwell time (us)"] = pixel_time*1e6 #seconds to microseconds

        BF_angle = 1e3*max_angles["bf"]["end"] if not HAADF_inserted else 1e3*max_angles["haadf"]["start"]

        microscope_info["BF collection semi-angle (mrad)"]= float(np.round(BF_angle,1))
        microscope_info["ADF inner collection semi-angle (mrad)"] = float(np.round(1e3*max_angles["haadf"]["start"],1))
        microscope_info["ADF outer collection semi-angle (mrad)"] = float(np.round(1e3*max_angles["haadf"]["end"],1))

    if acquisition_type.lower() == "camera" or acquisition_type.lower() == "4d":
        camera_roi_mode = app.detectors.camera.get_roi()
        roi_name = camera_roi_mode.name
        microscope_info["Camera ROI Mode"] = roi_name
        if roi_name == "Lines_128":
            camera_pixels = (512,128)
            diffraction_scaling = 4
        if roi_name == "Lines_256":
            camera_pixels = (512, 256)
            diffraction_scaling = 2
        elif roi_name == "Disabled":
            camera_pixels = (512,512)
            diffraction_scaling = 1

        microscope_info["Diffraction semiangle (mrad)"] = float(np.round(camera_angle*1e3,2)/diffraction_scaling)
        microscope_info["Diffraction angle (mrad)"] = float(np.round(camera_angle*1e3*2,2)/diffraction_scaling)
        microscope_info["Camera pixel size (A^-1)"] = float(pixel_size_inv_angstrom),
        microscope_info["Rotation angle between diffraction pattern and stage XY (deg)"] = float(np.round(np.rad2deg(app.api.projection.get_camera_to_stage_rotation()),2))

        mrad_per_pixel = (microscope_info["Diffraction semiangle (mrad)"] * 2) / camera_pixels[1]  # angle calibration per pixel #TODO might not work for ROI mode...
        convergence_pixels = microscope_info["Convergence semiangle (mrad)"] / mrad_per_pixel  # convergence angle in pixels
        pixel_radius = convergence_pixels  # semi-angle and radius
        microscope_info["Dwell time (ms)"] = pixel_time*1e3 #seconds to milliseconds
        microscope_info["Predicted diffraction spot diameter (px)"] = float(np.round(pixel_radius,2)) #TODO check with ROI mode
        microscope_info["Camera acquisition size (px)"] = str(camera_pixels)
        microscope_info["Camera ROI mode"] = camera_roi_mode

    if use_precession==True:
        microscope_info["Precession enabled"]=use_precession
        microscope_info["precession angle (mrad)"] = app.api.scanning.get_precession_angle()*1e3
        microscope_info["precession angle (deg)"] = float(np.round(np.rad2deg(microscope_info["precession angle (mrad)"]/1000),2))
        microscope_info["Precession Frequency (kHz)"] = app.api.scanning.get_precession_frequency()/1e3

    if edx_enabled == True:
        edx_filter = app.api.xray.get_xray_filter_type()
        microscope_info["EDX detector filter"] = edx_filter.name
        if num_frames is not None:
            microscope_info["Number of frames"] = num_frames
            if scan_height is not None:
                microscope_info["Total scanning time (s)"] =num_frames*pixel_time*scan_width_px*scan_height
            else:
                microscope_info["Total scanning time (s)"] =num_frames*pixel_time*scan_width_px**2
            microscope_info["Total series dose (e-A-2)"] = microscope_info["Probe dose rate (e-A-2s-1)"]* microscope_info["Total scanning time (s)"]

    xy = app.stage.get_xy()
    z = app.stage.get_z()
    alpha = app.stage.get_alpha()
    beta = app.stage.get_beta()

    microscope_info["Alpha tilt (deg)"] = float(np.round(np.rad2deg(alpha),2))
    microscope_info["Beta tilt (deg)"] = float(np.round(np.rad2deg(beta),2))
    microscope_info["X (um)"] = float(np.round(xy[0]*1e6,2))
    microscope_info["Y (um)"] = float(np.round(xy[1]*1e6,2))
    microscope_info["Z (um)"] = float(np.round(z*1e6,2))

    return microscope_info


def downsample_diffraction(array4d,rescale_to=128, mode='sum'):#TODO tested ok
    """
    Down-sample the diffraction-pattern axes of a 4-D STEM block to reduce data size.
    The scan grid (first two axes) is untouched.

    Parameters
    ----------
    array4d : np.ndarray
        Input array of shape (Ny_scan, Nx_scan, Ny_dp, Nx_dp),
        where Ny_dp and Nx_dp are equal and a multiple of 128.
    mode : {'sum', 'mean'}, optional
        'sum'  → sum the intensities inside each block (default)
        'mean' → average the intensities inside each block.

    Returns
    -------
    np.ndarray
        Output array of shape (Ny_scan, Nx_scan, 128, 128) with the
        same dtype as the input.
    """
    if array4d.ndim != 4:
        raise ValueError("Expected a 4-D array (scan_y, scan_x, dp_y, dp_x)")

    ny_s, nx_s, ny_dp, nx_dp = array4d.shape
    # The reduction factor along each DP axis
    fy, fx = ny_dp // rescale_to, nx_dp // rescale_to

    if ny_dp % rescale_to or nx_dp % rescale_to:
        raise ValueError(
            f"Diffraction-pattern size must be a multiple of {rescale_to}; got "
            f"({ny_dp}, {nx_dp})."
        )
    if fy != fx:
        raise ValueError(
            "Diffraction pattern must be square (ny_dp == nx_dp)."
        )

    # Ensure a contiguous buffer so reshape is a zero-copy view
    a = np.ascontiguousarray(array4d)

    # (Ny_s, Nx_s, 128*fy, 128*fx)  ➜  (Ny_s, Nx_s, 128, fy, 128, fx)
    view = a.reshape(ny_s, nx_s, rescale_to, fy, rescale_to, fx)

    # Collapse the two small axes (fy, fx) in a single vectorised pass
    if mode == 'sum':
        return view.sum(axis=(-1, -3))
    elif mode == 'mean':
        return view.mean(axis=(-1, -3))
    else:
        raise ValueError("mode must be 'sum' or 'mean'")

def array2blo(data: np.ndarray,meta: dict,filename,intensity_scaling="crop",bin_data_to_128=True):
    """Save (Ny, Nx, Dy, Dx) array to *.blo* with correct RosettaSciIO keys."""

    try:
        import dask.array as da
        _is_dask = lambda x: isinstance(x, da.Array)
    except ModuleNotFoundError:
        _is_dask = lambda x: False

    if bin_data_to_128:
        data = downsample_diffraction(data,rescale_to=128,mode="sum")
        print("Data downscaled to 128")
    navigator = data.sum(axis=(2, 3))

    lower_percentile, upper_percentile = np.percentile(navigator, [1, 99])  # Get 1st and 99th percentiles
    clipped_image = np.clip(navigator, lower_percentile, upper_percentile)  # Clip to the range
    image_min = clipped_image.min()
    image_max = clipped_image.max()
    navigator = (255 * (clipped_image - image_min) / (image_max - image_min)).astype(np.uint8)

    if data.ndim != 4:
        raise ValueError("`data` must be 4-D (Ny, Nx, Dy, Dx)")
    Ny, Nx, Dy, Dx = map(int, data.shape)

    px_nm = meta["Pixel size (nm)"]

    axes = [
        {"name": "scan_y", "units": "nm", "index_in_array": 0,
         "size": Ny, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "scan_x", "units": "nm", "index_in_array": 1,
         "size": Nx, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "qy", "units": "px", "index_in_array": 2,
         "size": Dy, "offset": 0.0, "scale": 1.0, "navigate": False},
        {"name": "qx", "units": "px", "index_in_array": 3,
         "size": Dx, "offset": 0.0, "scale": 1.0, "navigate": False},
    ]

    signal = {
        "data": data,
        "axes": axes,
        "metadata": {"acquisition": meta},
        "original_metadata": meta,
        "attributes": {"_lazy": _is_dask(data)},
    }

    file_writer(
        filename,
        signal,
        intensity_scaling=intensity_scaling,
        navigator=navigator,
        endianess="<",
        show_progressbar=True,
    )
    return #TODO tested ok

def dose_budget(budget_el_per_ang=10,probe_current=None,beam_size=None): #TODO test maths

    app = get_app

    dose_budget = budget_el_per_ang #electrons_per_square_angstrom
    if probe_current is None:
        probe_current = app.api.illumination.get_current() #amps

    electrons_per_amp = 1 / scipy.constants.elementary_charge
    electrons_in_probe = electrons_per_amp * probe_current  # electrons in probe per second

    if beam_size is None:
        beam_size = app.api.illumination.get_beam_diameter() #get current probe size
    beam_area = np.pi * (beam_size / 2) ** 2  # assume circular probe
    beam_dose_rate = electrons_in_probe/beam_area #electrons_per_square_meter

    beam_dose_rate_angstroms = beam_dose_rate/1e20
    max_exposure_time_seconds = dose_budget/beam_dose_rate_angstroms
    min_FPS = 1/max_exposure_time_seconds

    results = {"Acquisition dose rate,e-a-2s-1": float(np.round(beam_dose_rate_angstroms,1)),"Maximum acquisition time (s)": max_exposure_time_seconds,"Maximum acquisition time (us)": float(np.round(max_exposure_time_seconds*1e6,2)), "Minimum camera frequency (FPS)":float(np.round(min_FPS,1))}

    return results

def read_tiff_series(folder_path, sort=True, return_metadata=False):
    """
    Reads all TIFF images from a folder into a list of NumPy arrays.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing TIFF images.
    sort : bool, optional
        If True (default), sorts files alphabetically before reading.
    return_metadata : bool, optional
        If True, returns a list of (image, metadata) tuples.
        Otherwise, returns only image arrays.

    Returns
    -------
    list of np.ndarray
        List of 2D or 3D image arrays (depending on the data).
        If return_metadata=True, returns list of (array, metadata) tuples.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no TIFF files.
    """

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Collect all .tif / .tiff files
    tiff_files = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.tif', '.tiff'))]

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {folder_path}")

    if sort:
        tiff_files.sort()

    images = []
    for filename in tiff_files:
        path = os.path.join(folder_path, filename)
        with tiff.TiffFile(path) as tif:
            img = tif.asarray()
            if return_metadata:
                metadata = tif.imagej_metadata or tif.pages[0].tags
                images.append((img, metadata))
            else:
                images.append(img)

    return images

def view_all_models(host=None):
    if host == None:
        host = config.inference.host

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host, image_encoder='.tiff')
    models = manager.list_all_models() #gets all of the available models
    descriptions = []
    for i in models:
        description = manager.describe_model(i)
        print(f"{description[0]["modelName"]}, Version {description[0]["modelVersion"]}, Current workers available {description[0]["minWorkers"]}, Processor type {description[0]["deviceType"]}")
        descriptions.append((description[0]["modelName"],description[0]["minWorkers"]))
    return descriptions

def model_has_workers(model_name,host=None):

    if host == None:
        host = config.inference.host

    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                    image_encoder='.tiff')  # contacts the model manager
    model_status = manager.describe_model(model_name)
    if model_status[0]["minWorkers"] != 0:
        has_workers = True
    else: has_workers = False

    return has_workers


def crop_images_to_fixed_square(image_rows, square_size):
    """
    Crop a nested list of images (as NumPy arrays) to a fixed square size, centered around the middle
    of each image, processing the entire list at once.

    :param image_rows: Nested list of NumPy array images (rows of images)
    :param square_size: The fixed size of the square to crop (int)
    :return: Nested list of cropped NumPy array images (rows of cropped images)
    """
    cropped_rows = []  # List to store cropped rows
    while image_rows:
        # Remove the first row from the list
        row = image_rows.pop(0)
        # Assume all images in the row have the same dimensions (use the first image)
        height, width = row[0].shape[:2]
        # Validate the square size once for the row
        if square_size > height or square_size > width:
            raise ValueError(
                f"Square size {square_size} is too large for images with dimensions ({height}, {width})")
        # Calculate the top-left corner once for the row
        top = (height - square_size) // 2
        left = (width - square_size) // 2
        # Process the row
        cropped_row = []
        for img in row:
            cropped_img = img[top:top + square_size, left:left + square_size]# Crop the image using the precomputed coordinates
            cropped_row.append(cropped_img)# Add the processed row to the cropped_rows list
        cropped_rows.append(cropped_row)

    return cropped_rows

def crop_center_square(data: np.ndarray, square_size: int, copy: bool = False) -> np.ndarray:
    """
    Center-crop the last two axes (Y, X) of an array to a fixed square.

    Works for:
      - (camY, camX)
      - (N, camY, camX)
      - (scanY, scanX, camY, camX)
      - any shape ending in (..., camY, camX)

    Parameters
    ----------
    data : np.ndarray
        Input array with at least 2 dimensions.
    square_size : int
        Target square size.
    copy : bool
        If True, return a contiguous copy. If False (default), return a view.

    Returns
    -------
    np.ndarray
        Cropped array.
    """
    if data.ndim < 2:
        raise ValueError(f"data must have at least 2 dims, got {data.ndim}")

    camY, camX = data.shape[-2], data.shape[-1]
    if square_size > camY or square_size > camX:
        raise ValueError(
            f"square_size={square_size} exceeds last-two dims ({camY}, {camX})"
        )

    top = (camY - square_size) // 2
    left = (camX - square_size) // 2

    cropped = data[..., top:top + square_size, left:left + square_size]

    if copy:
        # Make it contiguous and independent of the original buffer
        return np.ascontiguousarray(cropped)

    return cropped