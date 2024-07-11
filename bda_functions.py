import numpy as np
from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT
from time import sleep
from tqdm import tqdm

#from utilities import get_microscope_parameters,get_number_of_nav_pixels,calculate_dose,create_circular_mask


def scan_4D_tool(scan_width_px=128,camera_frequency_hz=4500,use_precession=False):
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


def acquire_datapoint(num_pixels,dwell_time_s,output="sum",use_precession=False): #checked ok
    camera_frequency_hz = 1/dwell_time_s
    point = scan_4D_tool(num_pixels,camera_frequency_hz,use_precession)
    if output == "sum":
        output_data = np.sum(np.asarray(point,dtype=np.uint64)) #sums 4D acquisition to single diffraction pattern
    else:
        output_data = point #gives a list of length scanX*scanY
    return output_data



def beam_size_matched_acquisition(pixels=32,dwell_time_s=1e-3,output="sum",precession=False): #checked ok
    #optical_mode = grpc_client.microscope.get_optical_mode()

    beam_size = grpc_client.illumination.get_beam_diameter() #in meters

    matched_sampling_fov = beam_size*pixels #calculates FOV for 1:1 beam:pixel sampling
    grpc_client.scanning.set_field_width(matched_sampling_fov) #set fov in meters
    diffraction_data = acquire_datapoint(pixels,dwell_time_s,output=output,use_precession=precession) #acquires a 4D-dataset and sums to 1 diffraction pattern
    return diffraction_data



