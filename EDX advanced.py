import os
import threading

from stem_measurements import shift_measurements
from PyQt5.QtWidgets import QApplication
import pickle
from time import sleep
import numpy as np
import easygui as g
import cv2 as cv2
from tqdm import tqdm
from sys import getsizeof
import matplotlib.colors as mcolors
from collections import defaultdict
import xraydb as xdb
import scipy.signal
import matplotlib.pyplot as plt
from grid_strategy import strategies
from PIL import Image
from serving_manager.api import registration_model
from stem_measurements import edx_processing

from expert_pi.__main__ import window #TODO something is not right with this on some older versions of expi
from expert_pi import grpc_client
from expert_pi.app import scan_helper
from expert_pi.grpc_client.modules._common import DetectorType as DT, CondenserFocusType as CFT,RoiMode as RM
from serving_manager.api import TorchserveRestManager
from expert_pi.app import app
from expert_pi.gui import main_window

window = main_window.MainWindow()
controller = app.MainApp(window)
cache_client = controller.cache_client

from expert_pi.RSTEM.utilities import generate_colorlist,generate_colormaps,collect_metadata

host_F4 = ""
host_P3 = "172.20.32.1" #TODO confirm
host_P2 = ""
host_global = '172.16.2.86'


host = host_global

def acquire_EDX_map_core(frames=100,pixel_time=5e-6,fov=None,scan_rotation=0,num_pixels=1024,drift_correction_method="patches",verbose_logging=False,overscan=0):
    """Parameters
    frames: number of scans
    pixel_time: in seconds
    fov: in microns
    scan_rotation in degrees
    num_pixels: scan dimensions
    drift_correction_method: either "patches" for openCV template matching, "ML" uses trained AI drift correction
    verbose_logging: reports the shifts at each tilt
    overscan: is the percentage overscan for passive drift correction"""

    folder = g.diropenbox("Select folder to save mapping layers into")
    file_name = "\\EDX_map_frame"
    print("Predicted measurement time:", (pixel_time*num_pixels**2*frames/60)*1.2, "min")

    overscan_percentage = 1+(overscan/100)
    #print(f"overscanning by {overscan}%")


    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                 [-np.sin(scan_rotation), np.cos(scan_rotation)]])

    grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation))
    if fov is None:
        fov=grpc_client.scanning.get_field_width() #in meters
    if fov is not None:
        grpc_client.scanning.set_field_width(fov*overscan_percentage) #in meters

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is True and grpc_client.stem_detector.get_is_inserted(
                DT.HAADF) is False:
        tracking_signal = "BF"
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        tracking_signal = "BF"
        grpc_client.projection.set_is_off_axis_stem_enabled(True) #use off axis BF for tracking if both detectors are out
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        tracking_signal = "HAADF"
    if grpc_client.stem_detector.get_is_inserted(DT.BF) is True and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        tracking_signal = "HAADF"
    print(f"Image tracking using {tracking_signal} images")

    num_pixels = num_pixels*overscan_percentage

    map_data = []
    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
    header, data = cache_client.get_item(scan_id, num_pixels**2)
    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)
    map_data.append((header,data,initial_shift))

    with open(f"{folder}{file_name}_0.pdat", "wb") as f:
        pickle.dump(map_data[0], f) #first item from map data list of scans

    image_list = []
    anchor_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)
    image_list.append(anchor_image)

    for i in range(1, frames):
        print(f"Acquiring frame {i} of {frames}")
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1, detectors=[DT.BF, DT.HAADF, DT.EDX1, DT.EDX0])
        header, data = cache_client.get_item(scan_id, num_pixels**2)
        current_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

        with open(f"{folder}{file_name}_{i}.pdat", "wb") as f:
            pickle.dump((header, data, current_shift), f)

        # apply drift correction between images:
        series_image = data["stemData"][tracking_signal].reshape(num_pixels, num_pixels)
        shift_offset = drift_correction_core(fov,anchor_image,series_image,test_methods=False,method=drift_correction_method)
        grpc_client.illumination.set_shift({"x": current_shift['x'] - shift_offset[0], "y": current_shift['y'] - shift_offset[1]}, grpc_client.illumination.DeflectorType.Scan) #TODO check if it should be - or + shifts

        image_list.append(series_image)
        map_data.append((header,data,current_shift))
        if verbose_logging == True: #TODO untested
            print(f"Frame {i}, X shift {current_shift['x']*1e9 - shift_offset[0]*1e9} nm, Y shift {current_shift['y']*1e9 - shift_offset[1]*1e9} nm")
            print("Map data RAM usage",getsizeof(map_data)*1e-6,"Megabytes")
            print("Image list RAM usage",getsizeof(image_list)*1e-6,"Megabytes")
            #image_view.set_data(series_image) #updates image plot
            #ax1.scatter(shift_offset[0],shift_offset[1])
            #fig.canvas.draw()
            #QApplication.processEvents() #sends command to backend to update plots

    """Add in write metadata function"""
    metadata = collect_metadata(acquisition_type="EDX",scan_width_px=num_pixels,use_precession=False,pixel_time=pixel_time,scan_rotation=scan_rotation,edx_enabled=True,camera_pixels=512,num_frames=frames)


    return map_data,metadata



def drift_correction_core(fov,anchor_image,series_image,test_methods=True,method=None):
    """Core to test different drift correction methods"""
    drift_correction_methods = ["ML","Patches","Cross correlation",None]

    measured_offset = []
    confidences = []

    if test_methods == True:
        for dc_method in drift_correction_methods:
            if dc_method == "ML":
                registration = registration_model(np.concatenate([anchor_image, series_image], axis=1),
                                                  'TEMRegistration', host=host, port='7443',
                                                  image_encoder='.tiff')
                raw_shift = registration[0]["translation"]
                shift_offset = (raw_shift[0]*fov, raw_shift[1]*fov)
                measured_offset.append(shift_offset)
                confidence = 0
                confidences.append(confidence)
            if dc_method == "Patches":
                shift_offset,confidence = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov,
                                                                         method=shift_measurements.Method.PatchesPass2,
                                                                                    get_corr_coeff=True)
                measured_offset.append(shift_offset)
            if dc_method == "Cross corellation":
                shift_offset,confidence = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov,
                                                                         method=shift_measurements.Method.CrossCorr,
                                                                                    get_corr_coeff=True)
                measured_offset.append(shift_offset)
                confidences.append(confidence)
            if dc_method == None:
                shift_offset = (0, 0)
                measured_offset.append(shift_offset)
                confidence = 0
                confidence.append(confidence)

        return measured_offset,confidences,drift_correction_methods


    if method is "ML":
        registration = registration_model(np.concatenate([anchor_image, series_image], axis=1),
                                          'TEMRegistration', host=host, port='7443',
                                          image_encoder='.tiff')
        raw_shift = registration[0]["translation"]
        shift_offset = (raw_shift[0]*fov, raw_shift[1]*fov)
        measured_offset.append(shift_offset)
    elif method is "patches":
        """Patches pass 2"""
        shift_offset = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov,
                                                                 method=shift_measurements.Method.PatchesPass2)
        measured_offset.append(shift_offset)
    elif method is "xcorr":
        """Cross correlation"""
        shift_offset = shift_measurements.get_offset_of_pictures(anchor_image, series_image, fov,
                                                  method=shift_measurements.Method.CrossCorr)
        measured_offset.append(shift_offset)
    elif method is None:
        shift_offset = (0,0)
        measured_offset.append(shift_offset)

    return measured_offset