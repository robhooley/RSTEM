from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import numpy as np
import scipy.constants
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
        shape_4D = image_array.shape #shape of dataset
        dp_shape = shape_4D[2],shape_4D[3] #number of pixels in DP
        print("Metadata is present")
        convergence_semiangle = metadata.get("Convergence semiangle (mrad)")
        diffraction_angle = metadata.get("Diffraction semiangle (mrad)")
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



#TODO test new metadata function
def get_microscope_parameters(scan_width_px=None,use_precession=False,camera_frequency_hz=None,STEM_dwell_time=None,scan_rotation=None):
    """Extracts acquisition conditions from the microscope and stores them in a dictionary
    Parameters:
    scan_width_px: the number of pixels in the x axis
    use_precession: True or False
    camera_frequency_hz: The camera acquisition rate in FPS or Hz
    """
    if scan_width_px is None:
        scan_width_px = get_number_of_nav_pixels()

    fov = grpc_client.scanning.get_field_width() #get the current scanning FOV
    pixel_size_nm = (fov/scan_width_px)*1e9 #work out the pixel size in nanometers
    time_now = datetime.now()
    acquisition_time = time_now.strftime("%d_%m_%Y %H_%M")
    energy=grpc_client.gun.get_high_voltage()
    wavelength = calculate_wavelength(energy)
    pixel_size_inv_angstrom = (2*grpc_client.projection.get_max_camera_angle())*1e-3/(
                wavelength*0.01)/512 #assuming 512 pixels

    camera_dwell_time = 1/camera_frequency_hz*1e3

    if camera_frequency_hz == None:
        dwell_time_units = "us"
        dwell_time = STEM_dwell_time
        dwell_time_seconds = dwell_time*1e-6
        max_angles = grpc_client.projection.get_max_detector_angles()
        HAADF_inserted = grpc_client.stem_detector.get_is_inserted(grpc_client.stem_detector.DetectorType.HAADF)
        bf_max = 1e3*max_angles["bf"]["end"] if not HAADF_inserted else 1e3*max_angles["haadf"]["start"]
        ADF_min = 1e3*max_angles["haadf"]["start"]
        ADF_max = 1e3*max_angles["haadf"]["end"]
    else:
        dwell_time_units = "ms"
        dwell_time = camera_dwell_time
        dwell_time_seconds = dwell_time/1e3



    microscope_info = { #creates a dictionary of microscope parameters
    "Acquisition date and time":acquisition_time,
    "High Tension (kV)":energy/1e3,
    "Probe current (pA)" : grpc_client.illumination.get_current()*1e12,
    "Convergence semiangle (mrad)" : grpc_client.illumination.get_convergence_half_angle()*1e3,
    "Beam diameter (d50) (nm)" : grpc_client.illumination.get_beam_diameter()*1e9,"Scan width (px)":scan_width_px,
    "FOV (um)" : fov*1e6,
    "Pixel size (nm)" : pixel_size_nm,
    "Scan width (px)":scan_width_px,
    f"Dwell time ({dwell_time_units})": dwell_time,
    "Dwell time (s)":dwell_time_seconds,
    "Diffraction semiangle (mrad)" : grpc_client.projection.get_max_camera_angle()*1e3,
    "Camera pixel size (A^-1)":pixel_size_inv_angstrom,
    "Scan rotation (deg)":scan_rotation,
    "Rotation angle between diffraction pattern and stage XY (deg)":np.rad2deg(grpc_client.projection.get_camera_to_stage_rotation()),
    "Alpha tilt (deg)": grpc_client.stage.get_alpha()/np.pi*180,
    "Beta tilt (deg)": grpc_client.stage.get_beta()/np.pi*180,
    "X (mm)": grpc_client.stage.get_x_y()["x"]*1e3,
    "Y (mm)": grpc_client.stage.get_x_y()["y"]*1e3,
    "Z (mm)": grpc_client.stage.get_z()*1e3,
    }
    if use_precession is True:
        microscope_info["Precession enabled"]=True
        microscope_info["precession angle (mrad)"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["Precession Frequency (kHz)"] = grpc_client.scanning.get_precession_frequency()/1e3
    if camera_frequency_hz == None:
        microscope_info["BF collection semi-angle (mrad)"]= bf_max
        microscope_info["ADF inner collection semi-angle (mrad)"] = ADF_min
        microscope_info["ADF outer collection semi-angle (mrad)"] = ADF_max

    return microscope_info

def calculate_dose_from_ui_old(): # TODO deprecate
    """This extracts the information from the ExpertPI interface and calculates the dose"""
    illumination_parameters = grpc_client.illumination.get_current()
    probe_current = illumination_parameters["current"] #in amps

    dwell_time = window.scanning.pixel_time_spin.value() #always in us from expi window
    electrons_per_amp = 1/scipy.constants.elementary_charge
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
    if pixel_to_probe_ratio > 1:
        #print(f"Pixel size is {pixel_to_probe_ratio} times larger than probe size, undersampling conditions")
        sampling_conditions = "Undersampling"
    elif pixel_to_probe_ratio < 1 :
        #print(f"Probe size is {pixel_to_probe_ratio} times larger than probe size, oversampling conditions")
        sampling_conditions = "Oversampling"
    else:
        #print("Probe size and pixel sizre are perfectly matched")
        sampling_conditions = "Perfect sampling"

    dose_values = {"Pixel size":pixel_size,
                   "Probe size": probe_size,
    "Pixel dose e-nm-2": electrons_per_meter_square_pixel*1e-18,
    "Probe dose e-nm-2" :electrons_per_meter_square_probe*1e-18,
    "Pixel dose e-A-2": electrons_per_meter_square_pixel*1e-20,
    "Probe dose e-A-2" :electrons_per_meter_square_probe*1e-20,
    "Pixel dose e-m-2": electrons_per_meter_square_pixel,
    "Probe dose e-m-2":electrons_per_meter_square_probe,"Sampling conditions":sampling_conditions}

    return dose_values #returns dose values and the unit

def calculate_dose_old_metadata(metadata,num_pixels): #TODO uses old metadata format
    probe_current = metadata["probe current in picoamps"]*1e-12 #in amps

    dwell_time = metadata["Dwell time in ms"]
    electrons_per_amp = 1/scipy.constants.elementary_charge
    electrons_in_probe = electrons_per_amp*probe_current #electrons in probe per second
    electrons_per_pixel_dwell = electrons_in_probe*(dwell_time/1e3) #divide by dwell time in seconds

    """pixel size calculation"""
    scan_fov = metadata["FOV in microns"]*1e-6

    pixel_size = scan_fov/num_pixels #in meters
    pixel_area = pixel_size**2
    electrons_per_meter_square_pixel = electrons_per_pixel_dwell/pixel_area

    """probe size calculation"""
    probe_size = metadata["beam diameter (d50) in nanometers"]*1e-9 #in meters
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
        print("Probe size and pixel size are perfectly matched")
        sampling_conditions = "Perfect sampling"

    pixel_dose = electrons_per_meter_square_pixel*1e-18
    probe_dose = electrons_per_meter_square_probe*1e-18

    return (probe_dose,pixel_dose) #returns dose values and the unit

def calculate_dose(metadata=None): #TODO test this, can deprecate calculate_dose_fom_ui
    """Returns a dictionary contaning the calculated dose for the probe size and the pixel size in several units
    This requires only the metadata dictionary for a particular acquisition
    If the metadata is not provided, it will take the current state of the microscope and use that"""

    if metadata is None:
        current_state = get_microscope_parameters()
        probe_current = current_state["Probe current (pA)"]*1e-12  # in amps
        scan_fov = current_state["FOV um"]*1e-6
        dwell_time_seconds = current_state["Dwell time (s)"]
        print("Taking dwell time from UI window, may be different than acquisition conditions")
        probe_size = current_state["Beam diameter (d50) (nm)"]*1e-9  # in meters
        num_pixels = current_state["Scan width (px)"]
        print("Taking number of pixels from UI window, may be different than acquisition conditions")

    else:
        probe_current = metadata["Probe current (pA)"]*1e-12 #in amps
        scan_fov = metadata["FOV um"]*1e-6
        dwell_time_seconds = metadata["Dwell time (s)"]
        probe_size = metadata["Beam diameter (d50) (nm)"]*1e-9 #in meters
        num_pixels = metadata["Scan width (px)"]

    electrons_per_amp = 1/scipy.constants.elementary_charge
    electrons_in_probe = electrons_per_amp*probe_current #electrons in probe per second
    electrons_per_pixel_dwell = electrons_in_probe*(dwell_time_seconds) #divide by dwell time converted to seconds

    """pixel size calculation"""


    pixel_size = scan_fov/num_pixels #in meters
    pixel_area = pixel_size**2
    electrons_per_meter_square_pixel = electrons_per_pixel_dwell/pixel_area

    """probe size calculation"""

    probe_area = np.pi*(probe_size/2)**2 #assume circular probe
    electrons_per_meter_square_probe = electrons_per_pixel_dwell/probe_area

    """Calculate pixel size to probe size ratio"""
    pixel_to_probe_ratio = pixel_size/probe_size
    print(pixel_to_probe_ratio)
    if pixel_to_probe_ratio > 1:
        #print(f"Pixel size is {pixel_to_probe_ratio} times larger than probe size, undersampling conditions")
        sampling_conditions = "Undersampling"
    elif pixel_to_probe_ratio < 1 :
        #print(f"Probe size is {pixel_to_probe_ratio} times larger than probe size, oversampling conditions")
        sampling_conditions = "Oversampling"
    else:
        #print("Probe size and pixel sizre are perfectly matched")
        sampling_conditions = "Perfect sampling"

    dose_values = {"Pixel size":pixel_size,
                   "Probe size": probe_size,
    "Pixel dose e-nm-2": electrons_per_meter_square_pixel*1e-18,
    "Probe dose e-nm-2" :electrons_per_meter_square_probe*1e-18,
    "Pixel dose e-A-2": electrons_per_meter_square_pixel*1e-20,
    "Probe dose e-A-2" :electrons_per_meter_square_probe*1e-20,
    "Pixel dose e-m-2": electrons_per_meter_square_pixel,
    "Probe dose e-m-2":electrons_per_meter_square_probe,"Sampling conditions":sampling_conditions}

    return dose_values #returns dose values and the unit

import matplotlib.transforms as tfrms

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



