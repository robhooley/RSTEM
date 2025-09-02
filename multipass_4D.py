from expert_pi.measurements import shift_measurements
#concept
"""Take a reference STEM image, then collect a 4D-dataset at max camera speed, drift correct using the virtual images
then acquire subsequent 4D-STEM acquisitions with inter-frame drift correction from the virtual images"""



"""After acquisition, sum the images together but only allow summation if the corresponding pixel is 99.9% similar by some metric, 
sum in enough frames to reach the required dwell time (2ms)"""

import numpy as np


from expert_pi import grpc_client
from serving_manager.api import registration_model

host = "172.27.153.166"

def frame_calculator(requested_time=2e-3,acquisition_time=1/4500):
    num_frames = int(requested_time/acquisition_time) #integer number of scans only
    return num_frames

def drift_correction_cycle(acquired_image, anchor_image,fov,scan_rotation):
    s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)  # get current beam shifts
    # apply drift correction between images:
    registration = registration_model(np.concatenate([anchor_image, acquired_image], axis=1),
                                      'TEMRegistration', host=host, port='7443',
                                      image_encoder='.tiff')  # measure offset of images
    raw_shift = registration[0]["translation"]
    real_shift_x = raw_shift[0] * fov  # shifts normalised between 0,1 proportion of image, convert to meters
    real_shift_y = raw_shift[1] * fov
    grpc_client.illumination.set_shift(
        {"x": s['x'] - real_shift_x, "y": s['y'] - real_shift_y},
        grpc_client.illumination.DeflectorType.Scan)  # apply shifts in microns to existing shifts#

def acquisition_handler(rectangle,frames):

    #initial_STEM =

    for i in range(0,frames):
        scan_number = i
        print(f"Scanning frame nuumber {i} of {frames}")
