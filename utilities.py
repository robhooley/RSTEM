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
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.widgets import Slider
import matplotlib.transforms as tfrms
from expert_pi import grpc_client
from expert_pi.__main__ import window

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
    predicted_dataset_size_with_buffer = predicted_dataset_size_gbytes*1.2
    free_ram = psutil.virtual_memory().free/1e9
    if verbose:
        print(f"There are {free_ram} Gb of RAM available,dataset predicted to be {predicted_dataset_size_with_buffer}Gb")
    #will_work = False
    if free_ram>predicted_dataset_size_with_buffer:
        will_work = True
    else:
        will_work = False
    return will_work

#TODO this needs to be split into sections by acquisition type and made into single layered dictionary
#TODO set sensible decimal places

def get_microscope_parameters(scan_width_px=None,use_precession=False,camera_frequency_hz=None,STEM_dwell_time=None,scan_rotation=0):
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

    if STEM_dwell_time == None: #only needed for acquisitions from the console that use the UI dwell time
        STEM_dwell_time = window.scanning.pixel_time_spin.value() #in us


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
        camera_dwell_time = 1/camera_frequency_hz*1e3
        dwell_time_units = "ms"
        dwell_time = camera_dwell_time
        dwell_time_seconds = dwell_time/1e3

    #TODO consider splitting it based on acquisition type?

    microscope_info = { #creates a dictionary of microscope parameters
    "Acquisition date and time":acquisition_time,
    "High Tension (kV)":energy/1e3,
    "Probe current (pA)" : grpc_client.illumination.get_current()*1e12,
    "Convergence semiangle (mrad)" : grpc_client.illumination.get_convergence_half_angle()*1e3,
    "Beam diameter (d50) (nm)" : grpc_client.illumination.get_beam_diameter()*1e9,
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

def calculate_FFT(image,fov=None): #TODO refactor to be less hacky
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    result = np.fft.fftshift(mag)

    cv2.log(result, result)
    result = cv2.addWeighted(result, alpha=65535/result.max(), src2=result, beta=0, gamma=0, dtype=cv2.CV_16U)
    if fov is not None:
        pixel_size = fov/image.shape[0]
        inverse_pixel_size = 1/pixel_size
        fig,ax = plt.subplots(1,1)
        extent_plt = result.shape[0]*inverse_pixel_size
        ax.imshow(result,extent=(-extent_plt/2, extent_plt/2, -extent_plt/2, extent_plt/2))
        plt.show()
    #TODO take field of view and make the FFT spatially calibrated
    return result

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

    if metadata is None: #TODO change to metadata
        current_state = get_microscope_parameters()
        probe_current = current_state["Probe current (pA)"]*1e-12  # in amps
        scan_fov = current_state["FOV (um)"]*1e-6
        dwell_time_seconds = current_state["Dwell time (s)"]
        #print("Taking dwell time from UI window, may be different than acquisition conditions")
        probe_size = current_state["Beam diameter (d50) (nm)"]*1e-9  # in meters
        num_pixels = current_state["Scan width (px)"]
        #print("Taking number of pixels from UI window, may be different than acquisition conditions")
        #TODO if metadata is none, give only probe dose rate no pixel size or total dose
    else:
        probe_current = metadata["Probe current (pA)"]*1e-12 #in amps
        scan_fov = metadata["FOV (um)"]*1e-6
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
    if pixel_to_probe_ratio > 1:
        #print(f"Pixel size is {pixel_to_probe_ratio} times larger than probe size, undersampling conditions")
        sampling_conditions = "Undersampling"
        #print("Reccomended to use probe size calculation")
        reccomended = "Probe size calculation"
    elif pixel_to_probe_ratio < 1 :
        #print(f"Probe size is {pixel_to_probe_ratio} times larger than probe size, oversampling conditions")
        sampling_conditions = "Oversampling"
        #print("Reccomended to use pixel size calculation")
        reccomended = "Pixel size calculation"
    else:
        #print("Probe size and pixel size are perfectly matched")
        sampling_conditions = "Perfect sampling"

    dose_rate_probe_angstroms = electrons_per_meter_square_probe*1e-20/dwell_time_seconds
    dose_rate_pixel_angstroms = electrons_per_meter_square_pixel*1e-20/dwell_time_seconds

    dose_values = {"Pixel size":pixel_size,
                   "Probe size": probe_size,
                   "Probe current (pA)":probe_current,
    "Pixel dose e-nm-2": electrons_per_meter_square_pixel*1e-18,
    "Probe dose e-nm-2" :electrons_per_meter_square_probe*1e-18,
    "Pixel dose e-A-2": electrons_per_meter_square_pixel*1e-20,
    "Probe dose e-A-2" :electrons_per_meter_square_probe*1e-20,
    "Pixel dose e-m-2": electrons_per_meter_square_pixel,
    "Probe dose e-m-2":electrons_per_meter_square_probe,
    "Pixel dose rate e-A-2s-1":dose_rate_pixel_angstroms,
    "Probe dose rate-A-2s-1":dose_rate_probe_angstroms,
    "Sampling conditions":sampling_conditions,
                   "Reccomended calculation":reccomended}

    return dose_values #returns dose values and the unit

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

def import_tiff_series_basic(scan_width=None):
    """Loads in a folder of TIFFs and creates a 4D-array for use with other functions"""
    directory = g.diropenbox("Select directory","Select Directory")
    if scan_width is None: #if scan width variable is empty, prompt user to enter it
        num_files = len(fnmatch.filter(os.listdir(directory), '*.tiff')) #counts how many .tiff files are in the directory
        guessed_scan_width = int(np.sqrt(num_files)) #assumes it is a square acquisition
        scan_width=g.integerbox(f"Enter scan width in pixels, there are {num_files} TIFF files in this folder, "
                                f"scan width might be {guessed_scan_width}","Enter scan width in pixels",
                                default=guessed_scan_width)

    scan_height = num_files/scan_width
    if scan_height ==int(scan_height):
        scan_height = int(scan_height)
    else:
        scan_width = g.integerbox(f"Enter scan width in pixels, previous entry was likely not correct, "
                                  f"there are {num_files} files in this folder width {scan_width},height {scan_height}", "Enter scan width in pixels",
                                  default=guessed_scan_width)

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

    return reshaped_array #just a data array reshaped to the 4D STEM acquisition shape

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

#TODO this needs to be split into sections by acquisition type and made into single layered dictionary
#TODO set sensible decimal places

def collect_metadata(acquisition_type,scan_width_px=None,use_precession=False,camera_frequency_hz=None,STEM_dwell_time=None,scan_rotation=0):
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
                wavelength*0.01)/512 #assuming 512 pixels #TODO change to camera pixels if in 4D mode

    if STEM_dwell_time == None: #only needed for acquisitions from the console that use the UI dwell time
        STEM_dwell_time = window.scanning.pixel_time_spin.value() #in us


    if camera_frequency_hz == None:
        dwell_time_units = "us"
        dwell_time = STEM_dwell_time
        dwell_time_seconds = dwell_time*1e-6
        max_angles = grpc_client.projection.get_max_detector_angles()
        HAADF_inserted = grpc_client.stem_detector.get_is_inserted(grpc_client.stem_detector.DetectorType.HAADF)

    else:
        camera_dwell_time = 1/camera_frequency_hz*1e3
        dwell_time = camera_dwell_time

    #TODO consider splitting it based on acquisition type?

    microscope_info = { #creates a dictionary of microscope parameters
    "Acquisition date and time":acquisition_time,
    "High Tension (kV)":energy/1e3,
    "Probe current (pA)" : grpc_client.illumination.get_current()*1e12,
    "Convergence semiangle (mrad)" : grpc_client.illumination.get_convergence_half_angle()*1e3,
    "Beam diameter (d50) (nm)" : grpc_client.illumination.get_beam_diameter()*1e9,
    "FOV (um)" : fov*1e6,
    "Pixel size (nm)" : pixel_size_nm,
    "Scan width (px)":scan_width_px,
    "Diffraction semiangle (mrad)" : grpc_client.projection.get_max_camera_angle()*1e3,
    "Diffraction angle (mrad)" : grpc_client.projection.get_max_camera_angle()*1e3*2,
    "Camera pixel size (A^-1)":pixel_size_inv_angstrom,
    "Scan rotation (deg)":scan_rotation,
    "Rotation angle between diffraction pattern and stage XY (deg)":np.rad2deg(grpc_client.projection.get_camera_to_stage_rotation()),
    }

    microscope_info["Acquisition type"]= acquisition_type

    if acquisition_type == "STEM":
        microscope_info["Dwell time (us)"] = dwell_time
        microscope_info["BF collection semi-angle (mrad)"]= 1e3*max_angles["bf"]["end"] if not HAADF_inserted else 1e3*max_angles["haadf"]["start"]
        microscope_info["ADF inner collection semi-angle (mrad)"] = 1e3*max_angles["haadf"]["start"]
        microscope_info["ADF outer collection semi-angle (mrad)"] = 1e3*max_angles["haadf"]["end"]

    if acquisition_type == "Camera":
        microscope_info["Dwell time (ms)"] = (1/(camera_frequency_hz/1000))
        microscope_info["Precession enabled"]=use_precession
        microscope_info["precession angle (mrad)"] = grpc_client.scanning.get_precession_angle()*1e3
        microscope_info["Precession Frequency (kHz)"] = grpc_client.scanning.get_precession_frequency()/1e3

        microscope_info["Predicted diffraction spot size"] = spot_radius_in_px()

    microscope_info["Alpha tilt (deg)"] = grpc_client.stage.get_alpha()/np.pi*180
    microscope_info["Beta tilt (deg)"] = grpc_client.stage.get_beta()/np.pi*180
    microscope_info["X (mm)"] = grpc_client.stage.get_x_y()["x"]*1e3
    microscope_info["Y (mm)"] = grpc_client.stage.get_x_y()["y"]*1e3
    microscope_info["Z (mm)"] = grpc_client.stage.get_z()*1e3

    return microscope_info

#TODO something in quibbler messes with Numba
def scrollable_plot(image_list,defocus_intervals):
    from pyquibbler import iquib, initialize_quibbler
    initialize_quibbler()
    def set_axes():
        shape = image_list[0].shape
        ax.set_aspect(1) #sets square aspect ratio for the plot
        ax.set_xlim(0, shape[0]) #axes limited to size of dataset with no excess
        ax.set_ylim(0, shape[1]) #axes limited to size of dataset with no excess

    # The function to be called anytime a slider's value changes
    def update(val):
        print("Updating")
        ax.clear() #clears the old data from the navigation plot
        set_axes() #rebuilds the axes
        ax.imshow(image_list[xposition.val],cmap="gray") #adds the new diffraction pattern
        if defocus_intervals:
            ax.set_title(f"Image {xposition.val}, defocus {defocus_intervals[xposition.val]} nm")
        else:
            ax.set_title("Image",xposition.val)
        #ax.set_title(("Image", int(xposition.val))) #Adds title to diffraction plot
        fig.canvas.draw_idle() #stops the interactive plotting

    # Define initial plotting space
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # builds a figure with 2 subplots
    set_axes()  # sets the axis scales
    ax.imshow(image_list[int(len(image_list)/2)],cmap="gray")
    ax.set_title(f"Image {int(len(image_list)/2)}, defocus {defocus_intervals[0]} nm")  # adds title to navigation image
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the position of the pattern in the x axis.
    xpos_allowed = np.arange(start=0, stop=len(image_list),
                             step=1)  # slider range is capped to integer number of pixels
    xpos = fig.add_axes([0.1, 0.1, 0.8, 0.03])  # size of slider in plot
    xposition = Slider(ax=xpos, label='Image', valmin=0, valstep=xpos_allowed, valmax=len(image_list),
                       valinit=int(len(image_list)/2))  # creates the slider

    xpos_ref = xposition.on_changed(update)

    plt.show(block=False)
