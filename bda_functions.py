import numpy as np
from time import sleep
from tqdm import tqdm
from expert_pi.__main__ import window #TODO something is not right with this on some older versions of expi
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, CondenserFocusType as CFT
from serving_manager.api import TorchserveRestManager
from expert_pi.app import app
from expert_pi.gui import main_window
window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client
import scipy
from expert_pi.RSTEM.utilities import get_microscope_parameters

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
        sampling_conditions = "Undersampling"
        reccomended = "Probe size calculation"
    elif pixel_to_probe_ratio < 1 :
        sampling_conditions = "Oversampling"
        reccomended = "Pixel size calculation"
    else:
        sampling_conditions = "Perfect sampling"
        reccomended = "Either"

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


#refactored 0.1.0
def scan_4D_tool(scan_width_px=64,camera_frequency_hz=4500,use_precession=False):
    """Parameters
    scan width: pixels
    camera_frequency: camera speed in frames per second up to 72000
    use_precession: True or False
    returns a list of images
    """

    if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(DT.HAADF) == True: #if either STEM detector is inserted
        grpc_client.stem_detector.set_is_inserted(DT.BF,False) #retract BF detector
        grpc_client.stem_detector.set_is_inserted(DT.HAADF, False) #retract ADF detector
        for i in tqdm(range(5),desc="stabilising after detector retraction",unit=""):
            sleep(1) #wait for 5 seconds
    grpc_client.projection.set_is_off_axis_stem_enabled(False) #puts the beam back on the camera if in off-axis mode
    sleep(0.2)  # stabilisation after deflector change
    scan_id = scan_helper.start_rectangle_scan(pixel_time=np.round(1/camera_frequency_hz, 8), total_size=scan_width_px, frames=1, detectors=[DT.Camera], is_precession_enabled=use_precession)
    image_list = [] #empty list to take diffraction data
    for i in range(scan_width_px): #retrives data one scan row at a time to avoid crashes
        header, data = cache_client.get_item(scan_id, scan_width_px)  # cache retrieval in rows
        camera_size = data["cameraData"].shape[1],data["cameraData"].shape[2] #gets shape of diffraction patterns
        for j in range(scan_width_px): #for each pixel in that row
            image_data = data["cameraData"][j] #take the data for that pixel
            image_data = np.asarray(image_data) #convers to numpy array
            image_data = np.reshape(image_data,camera_size) #reshapes data to an individual image
            image_list.append(image_data) #adds it to the list of images

    return image_list #tuple with image data and metadata


#refactored 0.1.0
def acquire_datapoint(num_pixels,dwell_time_s,output="sum",use_precession=False): #checked ok
    camera_frequency_hz = 1/dwell_time_s
    point = scan_4D_tool(num_pixels,camera_frequency_hz,use_precession)
    if output == "sum":
        output_data = np.sum(np.asarray(point,dtype=np.uint64),axis=0) #sums 4D acquisition to single diffraction pattern
    else:
        output_data = point #gives a list of length scanX*scanY
    return output_data

#refactored 0.1.0
def beam_size_matched_acquisition(pixels=8,dwell_time_s=1e-3,output="sum",precession=False): #checked ok
    #optical_mode = grpc_client.microscope.get_optical_mode()

    beam_size = grpc_client.illumination.get_beam_diameter() #in meters

    matched_sampling_fov = beam_size*pixels*2 #calculates FOV for 1:1 beam:pixel sampling #TODO check GRPC with Vojta
    grpc_client.scanning.set_field_width(matched_sampling_fov) #set fov in meters
    diffraction_data = acquire_datapoint(pixels,dwell_time_s,output=output,use_precession=precession) #acquires a 4D-dataset and sums to 1 diffraction pattern
    return diffraction_data



