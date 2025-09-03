from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os
import cv2 as cv2
import psutil
import fnmatch
from tqdm import tqdm
import easygui as g
import numpy as np
from time import sleep
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from rsciio.blockfile import file_writer

from matplotlib.widgets import Slider
import matplotlib.transforms as tfrms
#from expert_pi.__main__ import window

from expert_pi.__main__ import window #TODO something is not right with this on some older versions of expi
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, CondenserFocusType as CFT,RoiMode as RM
from serving_manager.api import TorchserveRestManager
from expert_pi.app import app
from expert_pi.gui import main_window

#from expert_pi.RSTEM.easy_4D_processing import scan_4D_basic


window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client


def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)
    mask = dist_from_center <= mask_radius
    return mask

def spot_radius_in_px(data_array):
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

def get_number_of_nav_pixels(): #checked ok
    scan_field_pixels = window.scanning.size_combo.currentText() #gets the string of number of pixels from the UI (messy)
    pixels = int(scan_field_pixels.replace(" px", "")) #replaces the px with nothing and converts to an integer
    return pixels

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
    """energy in electronvolts -> return in picometers"""
    phir = energy*(1 + scipy.constants.e*energy/(2*scipy.constants.m_e*scipy.constants.c**2))
    g = np.sqrt(2*scipy.constants.m_e*scipy.constants.e*phir)
    k = g/scipy.constants.hbar
    wavelength = 2*np.pi/k
    return wavelength*1e12  # to picometers

def generate_colorlist(num_colors_needed,mode=None):
    if mode == "Explore": #this is only used as a joke
        print("Using Explores colour palette")
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

def generate_colormaps(num_colors,num_bins=100,mode=None):
    colormaps = []
    color_list = generate_colorlist(num_colors,mode)
    for color in color_list:
        colors_ = [mcolors.to_rgb('black'), mcolors.to_rgb(color)]  #
        cmap_name = 'black_' + color
        colormaps.append(mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_, N=num_bins))

    return colormaps

def collect_metadata(acquisition_type=None,scan_width_px=None,use_precession=False,pixel_time=None,scan_rotation=0,edx_enabled=False,num_frames=None): #TODO needs testing
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

    if scan_width_px is None:
        scan_width_px = get_number_of_nav_pixels()

    fov = grpc_client.scanning.get_field_width() #get the current scanning FOV
    pixel_size_nm = (fov/scan_width_px)*1e9 #pixel size in nanometers
    time_now = datetime.now()
    acquisition_time = time_now.strftime("%d_%m_%Y %H_%M")
    energy=grpc_client.gun.get_high_voltage()
    wavelength = calculate_wavelength(energy)
    pixel_size_inv_angstrom = (2*grpc_client.projection.get_max_camera_angle())*1e-3/(
                wavelength*0.01)/512 #assuming 512 pixels

    probe_current = grpc_client.illumination.get_current() #amps
    probe_size = grpc_client.illumination.get_beam_diameter() #meters

    optical_mode = grpc_client.microscope.get_optical_mode()
    optical_mode = optical_mode.name

    max_angles = grpc_client.projection.get_max_detector_angles()
    max_camera_angle = grpc_client.projection.get_max_camera_angle()


    if pixel_time == None: #only needed for acquisitions from the console that use the UI dwell time
        pixel_time = window.scanning.pixel_time_spin.value() #in us

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
    "Probe current (pA)" : np.round(probe_current*1e12,2),
    "Convergence semiangle (mrad)" : np.round(grpc_client.illumination.get_convergence_half_angle()*1e3,2),
    "Beam diameter (d50) (nm)" : np.round(probe_size*1e9,2),
    "FOV (um)" : fov*1e6,
    "Pixel size (nm)" : np.round(pixel_size_nm,2),
    "Scan width (px)":scan_width_px,
    "Pixel time (s)": pixel_time,
    #"Diffraction semiangle (mrad)" : np.round(grpc_client.projection.get_max_camera_angle()*1e3,2),
    #"Diffraction angle (mrad)" : np.round(grpc_client.projection.get_max_camera_angle()*1e3*2,2),
    #"Camera pixel size (A^-1)":pixel_size_inv_angstrom,
    "Scan rotation (deg)":scan_rotation,
    "Pixel dose e-nm-2": electrons_per_meter_square_pixel * 1e-18,
    "Probe dose e-nm-2": electrons_per_meter_square_probe * 1e-18,
    "Pixel dose e-A-2": electrons_per_meter_square_pixel * 1e-20,
    "Probe dose e-A-2": electrons_per_meter_square_probe * 1e-20,
    "Pixel dose rate e-A-2s-1": dose_rate_pixel_angstroms,
    "Probe dose rate-A-2s-1": dose_rate_probe_angstroms,
    "Ratio of probe size to pixel size": pixel_to_probe_ratio}

    microscope_info["Acquisition type"]= acquisition_type

    if acquisition_type.lower() == "stem":
        HAADF_inserted = grpc_client.stem_detector.get_is_inserted(grpc_client.stem_detector.DetectorType.HAADF)
        microscope_info["Dwell time (us)"] = pixel_time*1e6 #seconds to microseconds
        microscope_info["BF collection semi-angle (mrad)"]= 1e3*max_angles["bf"]["end"] if not HAADF_inserted else 1e3*max_angles["haadf"]["start"]
        microscope_info["ADF inner collection semi-angle (mrad)"] = 1e3*max_angles["haadf"]["start"]
        microscope_info["ADF outer collection semi-angle (mrad)"] = 1e3*max_angles["haadf"]["end"]

    if acquisition_type.lower() == "camera" or "4d-stem":
        camera_roi_mode = grpc_client.scanning.get_camera_roi()
        camera_roi_mode = camera_roi_mode["roi_mode"].name
        if camera_roi_mode == "Lines_128":
            camera_pixels = (512,128)
            diffraction_scaling = 4
        if camera_roi_mode == "Lines_256":
            camera_pixels = (512, 256)
            diffraction_scaling = 2
        elif camera_roi_mode == "Disabled":
            camera_pixels = (512,512)
            diffraction_scaling = 1

        microscope_info["Diffraction semiangle (mrad)"] = np.round(max_camera_angle*1e3,2)/diffraction_scaling
        microscope_info["Diffraction angle (mrad)"] = np.round(max_camera_angle*1e3*2,2)/diffraction_scaling
        microscope_info["Camera pixel size (A^-1)"] = pixel_size_inv_angstrom,
        microscope_info["Rotation angle between diffraction pattern and stage XY (deg)"] = np.round(np.rad2deg(grpc_client.projection.get_camera_to_stage_rotation()),2),

        mrad_per_pixel = (microscope_info["Diffraction semiangle (mrad)"] * 2) / camera_pixels[1]  # angle calibration per pixel #TODO might not work for ROI mode...
        convergence_pixels = microscope_info["Convergence semiangle (mrad)"] / mrad_per_pixel  # convergence angle in pixels
        pixel_radius = convergence_pixels  # semi-angle and radius
        microscope_info["Dwell time (ms)"] = pixel_time*1e3 #seconds to milliseconds
        microscope_info["Predicted diffraction spot diameter (px)"] = np.round(pixel_radius,2) #TODO check with ROI mode
        microscope_info["Camera acquisition size (px)"] = str(camera_pixels)
        microscope_info["Camera ROI mode"] = camera_roi_mode

    if use_precession==True:
        microscope_info["Precession enabled"]=use_precession
        microscope_info["precession angle (mrad)"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["precession angle (deg)"] = np.round(np.rad2deg(microscope_info["precession angle (mrad)"]/1000),2)
        microscope_info["Precession Frequency (kHz)"] = grpc_client.scanning.get_precession_frequency()/1e3

    if edx_enabled == True:
        edx_filter = grpc_client.xray.get_xray_filter_type()
        microscope_info["EDX detector filter"] = edx_filter.name
        microscope_info["Number of frames"] = num_frames
        microscope_info["Total scanning time (s)"] =num_frames*pixel_time*scan_width_px*scan_width_px

    microscope_info



    microscope_info["Alpha tilt (deg)"] = np.round(grpc_client.stage.get_alpha()/np.pi*180,2)
    microscope_info["Beta tilt (deg)"] = np.round(grpc_client.stage.get_beta()/np.pi*180,2)
    microscope_info["X (um)"] = np.round(grpc_client.stage.get_x_y()["x"]*1e6,2)
    microscope_info["Y (um)"] = np.round(grpc_client.stage.get_x_y()["y"]*1e6,2)
    microscope_info["Z (um)"] = np.round(grpc_client.stage.get_z()*1e6,2)

    return microscope_info

def acquire_precession_tilt_series(upper_limit_degrees): #TODO untested
    filepath = g.diropenbox("Select directory to save series","Save location")
    beam_size = grpc_client.illumination.get_beam_diameter()

    grpc_client.scanning.set_scan_field_width(beam_size*2) # TODO check if this works
    pattern_list = []

    angle_list = list(np.linspace(0,upper_limit_degrees,0.1))


    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) == True:  # if either STEM detector is inserted
        grpc_client.stem_detector.set_is_inserted(DT.BF, False)  # retract BF detector
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False)  # retract ADF detector
        for i in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)  # wait for 5 seconds
    grpc_client.projection.set_is_off_axis_stem_enabled(
        False)  # puts the beam back on the camera if in off-axis mode
    sleep(0.2)  # stabilisation after deflector change

    scan_width_px = 8

    for i in angle_list:
        print(f"Current precession angle {i} degrees")
        radians = np.deg2rad(i)
        grpc_client.scanning.set_precession_angle(radians)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=1e-3,
                                                   total_size=scan_width_px, frames=1, detectors=[DT.Camera],
                                                   is_precession_enabled=True)
        image_list = []  # empty list to take diffraction data
        for i in tqdm(range(scan_width_px), desc="Retrieving data from cache", total=scan_width_px,
                      unit="chunks"):  # retrives data one scan row at a time to avoid crashes
            header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
            camera_size = data["cameraData"].shape[1], data["cameraData"].shape[2]  # gets shape of diffraction patterns
            image_data = data["cameraData"]  # take the data for that row
            image_row = np.reshape(image_data, (
            scan_width_px, camera_size[0], camera_size[1]))  # reshapes data to an individual image
            image_list.append(image_row)  # adds it to the list of images

        single_pattern = np.sum(np.asarray(image_list,dtype=np.uint64),axis=0)
        pattern_list.append(single_pattern)

    annotated_images = []
    for i in range(len(pattern_list)):
        image = pattern_list[i].astype(np.uint8)
        angle = angle_list[i]
        text = f"Precession angle {angle} degrees"
        cv2.putText(image, text, (10,10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(255, 225, 255))
        filename = filepath+f"\precession angle {angle} degrees.tiff"
        cv2.imwrite(filename,image)
        annotated_images.append(image)

    return image_list,angle_list

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

def dose_budget(budget_el_per_ang=10): #TODO untested

    dose_budget = budget_el_per_ang #electrons_per_square_angstrom
    probe_current = grpc_client.illumination.get_current() #amps

    electrons_per_amp = 1 / scipy.constants.elementary_charge
    electrons_in_probe = electrons_per_amp * probe_current  # electrons in probe per second

    probe_size = grpc_client.illumination.get_beam_diameter() #get current probe size
    probe_area = np.pi * (probe_size / 2) ** 2  # assume circular probe
    probe_dose_rate = electrons_in_probe/probe_area #electrons_per_square_meter

    probe_dose_rate_angstroms = probe_dose_rate/1e20
    max_exposure_time_seconds = dose_budget/probe_dose_rate_angstroms
    min_FPS = 1/max_exposure_time_seconds

    return probe_dose_rate_angstroms,max_exposure_time_seconds,min_FPS
