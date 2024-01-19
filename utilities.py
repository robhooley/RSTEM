import numpy as np
from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

#from expert_pi import grpc_client

def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None): #todo move to utilities file
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
        shape_4D = image_array.shape #shape of datasey
        dp_shape = shape_4D[2],shape_4D[3] #number of pixels in DP
        print("Metadata is present")
        convergence_semiangle = metadata.get("convergence semiangle mrad")
        diffraction_angle = metadata.get("diffraction semiangle in mrad") #TODO check if total or half angle
        mrad_per_pixel = diffraction_angle/dp_shape[0] #angle calibration per pixel
        convergence_pixels = convergence_semiangle/mrad_per_pixel #convergence angle in pixels
        pixel_radius = convergence_pixels #semi-angle and radius

    else:
        print("Metadata is not present")
        pixel_radius = 1

    return pixel_radius

def get_number_of_nav_pixels(): #checked ok
    scan_field_pixels = window.scanning.size_combo.currentText() #gets the string of number of pixels from the UI (messy)
    pixels = int(scan_field_pixels.replace(" px", "")) #replaces the px with nothing and converts to an integer
    return pixels

def get_microscope_parameters(scan_width_px,use_precession,camera_frequency_hz):
    """Extracts acquisition conditions from the microscope and stores them in a dictionary
    Parameters:
    scan_width_px: the number of pixels in the x axis
    use_precession: True or False
    camera_frequency_hz: The camera acquisition rate in FPS or Hz
    """
    fov = grpc_client.scanning.get_field_width() #get the current scanning FOV
    pixel_size_nm = (fov/scan_width_px)*1e9 #work out the pixel size in nanometers
    time_now = datetime.now()
    acquisition_time = time_now.strftime("%d_%m_%Y %H_%M")
    microscope_info = { #creates a dictionary of microscope parameters
    "scan width in pixels":scan_width_px,
    "FOV in microns" : fov*1e6,
    "pixel size nm" : pixel_size_nm,
    "probe current in picoamps" : grpc_client.illumination.get_current()*1e12,
    "convergence semiangle mrad" : grpc_client.illumination.get_convergence_half_angle()*1e3,
    "beam diameter (d50) in nanometers" : grpc_client.illumination.get_beam_diameter()*1e9,
    "diffraction semiangle in mrad" : grpc_client.projection.get_max_camera_angle()*1e3,
    "Dwell time in ms": (1/camera_frequency_hz)*1e3,
    "Acquisition date and time":acquisition_time
    }
    if use_precession is True:
        microscope_info["Using precssion"]="True"
        microscope_info["precession angle in mrad"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["Precession Frequency in HZ"] = grpc_client.scanning.get_precession_frequency()
    return microscope_info

def get_ui_dose_values(units="nm"): # TODO check on live instrument
    illumination_parameters = grpc_client.illumination.get_spot_size()
    probe_current = illumination_parameters["current"] #in amps

    dwell_time = window.scanning.pixel_time_spin.value() #always in us from expi window
    electrons_per_amp = 6.28e18
    electrons_in_probe = electrons_per_amp*probe_current #electrons in probe per second
    electrons_per_pixel_dwell = electrons_in_probe*(dwell_time/1e6) #divide by dwell time in seconds

    """pixel size calculation"""
    scan_fov = grpc_client.scanning.get_field_width()
    num_pixels = get_number_of_nav_pixels()
    pixel_size = scan_fov/num_pixels #in meters
    pixel_area = pixel_size**2
    electrons_per_meter_square_pixel = electrons_per_pixel_dwell/pixel_area

    """probe size calculation"""
    probe_size = illumination_parameters["d50"] #in meters
    probe_area = np.pi*(probe_size/2)**2 #assume circular probe
    electrons_per_meter_square_probe = electrons_per_pixel_dwell/probe_area

    """Calculate pixel size to probe size ratio"""
    pixel_to_probe_ratio = pixel_size/probe_size
    print(pixel_to_probe_ratio)
    if pixel_to_probe_ratio > 1:
        print(f"Pixel size is {pixel_to_probe_ratio} times larger than probe size, undersampling conditions")
        sampling_conditions = "Undersampling"
    elif pixel_to_probe_ratio < 1 :
        print(f"Probe size is {pixel_to_probe_ratio} times larger than probe size, oversampling conditions")
        sampling_conditions = "Oversampling"
    else:
        print("Probe size and pixel sizre are perfectly matched")
        sampling_conditions = "Perfect sampling"

    if units == "nm":
        pixel_dose = electrons_per_meter_square_pixel*1e-18
        probe_dose = electrons_per_meter_square_probe*1e-18

    elif units == "m":
        pixel_dose = electrons_per_meter_square_pixel
        probe_dose = electrons_per_meter_square_probe

    elif units == "A" or "angstrom" or "Angstrom":
        pixel_dose = electrons_per_meter_square_pixel*1e-20
        probe_dose = electrons_per_meter_square_probe*1e-20

    dose_unit = f"e-{units}-2"

    dose_values = {"pixel dose":pixel_dose,
                   "pixel size":pixel_size, #dictionary of values
                    "probe dose":probe_dose,
                   "probe size":probe_size,
                   "dose units":dose_unit,
                   "Sampling conditions":sampling_conditions}

    return dose_values #returns dose values and the unit

import matplotlib.transforms as tfrms

def create_scalebar(ax,scalebar_size_pixels,metadata):
    print(scalebar_size_pixels,"scalebar requested in pixels")
    pixel_size = metadata["pixel size nm"]*1e6
    scalebar_nm = int(scalebar_size_pixels*pixel_size)
    print("scalebar in nanometers",scalebar_nm)
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                        scalebar_nm,f"{scalebar_nm}nm", 'lower right',
                           color='white',
                           frameon=False,
                           fontproperties=fontprops)

    ax.add_artist(scalebar)

def create_plot_grid(num_fixed_images,num_variable_images):
    num_total_plots = num_fixed_images+num_variable_images
    if num_total_plots ==1 or 2 or 3:
        grid_rows, grid_cols = 1, num_total_plots #1x3 plot for 2 masks +1DP

    elif num_total_plots == 4:
        grid_rows, grid_cols = 2, 2 #2x2 plot for 3 masks +1DP
    elif num_total_plots == 5:
        grid_rows, grid_cols = 2, 3
 #TODO finish this mess