#TODO only verified functions
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from operator import itemgetter
import skimage.exposure
from scipy.integrate import simps
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from scipy.signal import find_peaks, savgol_filter
from skimage import restoration, feature, transform
import math
from math import sqrt
import pandas as pd
from collections import Counter
from serving_manager.api import TorchserveRestManager

from bisect import bisect_left
from expert_pi.RSTEM.easy_4D_processing import scan_4D_basic
from expert_pi.RSTEM.utilities import get_microscope_parameters, calculate_dose
#from expert_pi.RSTEM.utilities import get_microscope_parameters, calculate_dose
from expert_pi import grpc_client
from expert_pi.stream_clients import cache_client
from expert_pi.controllers import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT
#import json

#from utilities import get_microscope_parameters,get_number_of_nav_pixels,calculate_dose,create_circular_mask



def acquire_datapoint(num_pixels,dwell_time_s,output="sum",use_precession=False): #checked ok
    camera_frequency_hz = 1/dwell_time_s
    point = scan_4D_basic(num_pixels,camera_frequency_hz,use_precession)
    camera_data = point[0]
    if output == "sum":
        output_data = np.sum(camera_data,(0,1),dtype=np.float32) #sums 4D acquisition to single diffraction pattern
    else:
        output_data = camera_data #gives an array where shape is shape4D (scanX,scanY,cameraX,cameraY)
    return output_data



def beam_size_matched_acquisition(pixels=32,dwell_time_s=1e-3,output="sum",precession=False): #checked ok
    #optical_mode = grpc_client.microscope.get_optical_mode()

    beam_size = grpc_client.illumination.get_beam_diameter() #in meters

    matched_sampling_fov = beam_size*pixels #calculates FOV for 1:1 beam:pixel sampling
    grpc_client.scanning.set_field_width(matched_sampling_fov) #set fov in meters
    diffraction_data = acquire_datapoint(pixels,dwell_time_s,output=output,use_precession=precession) #acquires a 4D-dataset and sums to 1 diffraction pattern
    return diffraction_data,matched_sampling_fov



