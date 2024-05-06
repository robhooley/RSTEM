from expert_pi.__main__ import window
from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT
from expert_pi.grpc_client.modules._common import CondenserFocusType as CFT
from stem_measurements import shift_measurements
from tqdm import tqdm
import numpy as np
from time import sleep
import easygui as g
import cv2 as cv2
import fnmatch
import os
import json

from serving_manager.api import registration_model
from serving_manager.api import super_resolution_model
from serving_manager.api import TorchserveRestManager
from stem_measurements.shift_measurements import get_offset_of_pictures
from expert_pi.RSTEM.utilities import get_microscope_parameters
#registration_model(np.concatenate([original_image,translated_image],axis=1, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#registration_model(image, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#super_resolution_model(image=image, model_name='SwinIRImageDenoiser', host='172.19.1.16', port='7447') #for denoising
#manager = TorchserveRestManager(inference_port='8600', management_port='8081', host='172.19.1.16', image_encoder='.png')
#manager.infer(image=image, model_name='spot_segmentation') #spot detection
#manager.list_models()


host_F4 = ""
host_P3 = "172.20.32.1" #TODO confirm
host_P2 = ""
host_global = '172.16.2.86'


host = host_global

#TODO test with new rounded defocus offset numbers
def acquire_focal_series(extent_nm,steps=11,BF=True,ADF=False,num_pixels=1024,pixel_time_us=5):
    """Parameters
    extent_nm: Range the focal series should cover split equally around the current focus value
    steps: how many total steps the focal series should cover
    BF: True or False
    ADF: True or False
    num_pixels: default 1024
    pixel_time_us: default 5us"""

    current_defocus = grpc_client.illumination.get_condenser_defocus(CFT.C3)
    lower_defocus = current_defocus-(extent_nm*1e-9/2)
    higher_defocus = current_defocus+(extent_nm*1e-9/2)
    defocus_intervals = np.linspace(lower_defocus,higher_defocus,steps)

    image_series = []
    defocus_offsets = []
    print("Acquiring defocus series")
    for i in tqdm(defocus_intervals):

        grpc_client.illumination.set_condenser_defocus(i,CFT.C3) #get correct command
        defocus_now = grpc_client.illumination.get_condenser_defocus(CFT.C3)

        defocus_offset = current_defocus-defocus_now #this will be in meters
        defocus_offset_nm = defocus_offset*1e9
        defocus_offsets.append(np.round(defocus_offset_nm,decimals=1))

        scan_id = scan_helper.start_rectangle_scan(pixel_time=(pixel_time_us*1e-6), total_size=num_pixels, frames=1,
                                                   detectors=(DT.BF,DT.HAADF))
        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive small measurements in one chunk

        if BF==True and ADF==False:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            image_series.append(BF_image)
        if ADF==True and BF == False:
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append(ADF_image)
        if BF == True and ADF == True:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append((BF_image,ADF_image))

    return image_series,defocus_offsets

#TODO retest
def acquire_STEM( fov_um=None,pixel_time_us=None,num_pixels=1024,scan_rotation_deg=None):
    """Acquires a single STEM image from the requested detectors
    returns a tuple with the images and the metadata in a dictionary
    Parameters
    BF: True or False
    ADF: True or False
    fov : field of view in microns
    pixel_time_us: dwell time in microseconds
    num_pixels: scan width in pixels,
    scan_rotation: scan rotation in degrees """

    if scan_rotation_deg is not None:
        grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation_deg))
    if pixel_time_us is None:
        pixel_time = window.scanning.pixel_time_spin.value()*1e-6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us*1e-6
    if fov_um is not None:
        grpc_client.scanning.set_field_width(fov_um*1e-6) #in microns

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is False:
        grpc_client.projection.set_is_off_axis_stem_enabled(True) #ensure off axis BF is used if both detectors are out

    if grpc_client.stem_detector.get_is_inserted(DT.BF) is False and grpc_client.stem_detector.get_is_inserted(
            DT.HAADF) is True:
        grpc_client.projection.set_is_off_axis_stem_enabled(False)  # ensure off axis BF is used if both detectors are out
    metadata = get_microscope_parameters(scan_width_px = num_pixels,use_precession=False,camera_frequency_hz = None,
                                         STEM_dwell_time = pixel_time_us,scan_rotation=np.rad2deg(grpc_client.scanning.get_rotation()))

    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=[DT.BF,DT.HAADF])
    header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive small measurements in one chunk


    BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    return([BF_image,ADF_image],metadata)


#tested ok
def save_STEM(image,metadata=None,name=None,folder=None):
    """Parameters
    image: single array to be saved as a .tiff image
    metadata : optional dictionary to be saved as json
    name: optional user defined filename, otherwise will be called STEM
    folder: optional user defined folder, otherwise will show UI to select"""
    if folder == None:
        folder = g.diropenbox("Enter save location","Enter save location")
        folder + "\\"
    if name == None:
        num_files_in_dir = len(fnmatch.filter(os.listdir(folder), '*.tiff'))
        name = f"STEM000{num_files_in_dir+1}" #should increment the image number
    name = name+".tiff"
    filename = str(folder+"\\"+name)
    print(f"Saving {name} to {folder}")
    cv2.imwrite(filename,image)

    if metadata is not None:
        metadata_name = folder+"\\" + f"{name}_metadata.json"
        open_json = open(metadata_name, "w")
        json.dump(metadata, open_json, indent=6)
        open_json.close()



#tested ok without scan rotation
#TODO reintroduce scan rotation
def ML_drift_corrected_imaging(num_frames, pixel_time_us=None,num_pixels=None):
    """Parameters
    num_frames : integer number of frames to acquire
    pixel_time_us: pixel dwell time in microseconds
    num_pixels: number of pixels in scanned image
    Set the FOV and illumination conditions before calling the function, it will use whatever is set in the UI"""

    if pixel_time_us==None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    fov = grpc_client.scanning.get_field_width()

    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels ** 2) #retrive small measurements in one chunk

    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)
    #TODO this needs a tracking signal image flag to handle HAADF images with BF retracted
    for frame in range(num_frames):
        print("Acquiring frame",frame,"of",num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels) #reshape BF image
        BF_images_list.append(BF_image.astype(np.float64)) #add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels) #reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64)) #add to list

        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan) #get current beam shifts
        # apply drift correction between images:
        registration = registration_model(np.concatenate([BF_images_list[0],BF_image],axis=1),
                                          'TEMRegistration', host=host, port='7443',
                                          image_encoder='.tiff') #measure offset of images # TODO corrects to first image, should it correct to previous image?
        raw_shift = registration[0]["translation"]
        real_shift_x = raw_shift[0]*fov #shifts normalised between 0,1 proportion of image, convert to meters
        real_shift_y = raw_shift[1]*fov
        grpc_client.illumination.set_shift(
                {"x": s['x'] - real_shift_x , "y": s['y'] - real_shift_y },grpc_client.illumination.DeflectorType.Scan) #apply shifts in microns to existing shifts

    print("Post acquisition fine correction")
    aligned_BF_series,summed_BF,_ = align_series_ML(BF_images_list)
    aligned_ADF_series,summed_ADF,_ = align_series_ML(ADF_images_list)

    return (aligned_BF_series,summed_BF), (aligned_ADF_series, summed_ADF)

def acquire_series(num_frames, pixel_time_us=None,num_pixels=None ):
    if pixel_time_us == None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    print("Pixel time", int(pixel_time*1e9), "ns")
    fov = grpc_client.scanning.get_field_width()*1e6  # in microns
    print("Field of View in um", fov)

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []


    for frame in range(num_frames):
        print("Acquiring frame", frame, "of", num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)  # reshape BF image
        BF_images_list.append(BF_image.astype(np.float64))  # add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)  # reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64))  # add to list

    return BF_images_list
# tested ok without scan rotation
# TODO reintroduce scan rotation
#TODO untested
def ML_drift_corrected_imaging_with_scan_rotation(num_frames, pixel_time_us=None, num_pixels=None, scan_rotation=0):
    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                  [-np.sin(scan_rotation), np.cos(scan_rotation)]])

    if scan_rotation is not None:
        grpc_client.scanning.set_rotation(scan_rotation)

    if pixel_time_us == None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    fov = grpc_client.scanning.get_field_width()

    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                    detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels**2)  # retrive small measurements in one chunk

    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)

    for frame in range(num_frames):
        print("Acquiring frame", frame, "of", num_frames)
        scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                                   detectors=[DT.BF, DT.HAADF])

        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive image from cache
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)  # reshape BF image
        BF_images_list.append(BF_image.astype(np.float64))  # add to list
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)  # reshape ADF image
        ADF_images_list.append(ADF_image.astype(np.float64))  # add to list

        s = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)  # get current beam shifts
        # apply drift correction between images:
        registration = registration_model(np.concatenate([BF_images_list[-1], BF_image], axis=1),
                                          'TEMRegistration', host=host, port='7443',
                                          image_encoder='.tiff')  # measure offset of images
        raw_shift = registration[0]["translation"]
        real_shift_x = raw_shift[0]*fov  # shifts normalised between 0,1 proportion of image, convert to meters
        real_shift_y = raw_shift[1]*fov

        shift = np.dot(R, (real_shift_x,real_shift_y))  # rotate shifts back to scanning axes
        # print(i, (s0['x'] - s['x']) * 1e9, (s0['y'] - s['y']) * 1e9)
        grpc_client.illumination.set_shift(
                {"x": s['x'] + shift[0], "y": s['y'] + shift[1]},
                grpc_client.illumination.DeflectorType.Scan)  # apply shifts in microns to existing shifts

    print("Post acquisition correction")
    aligned_BF_series, summed_BF, _ = align_series_ML(BF_images_list)
    aligned_ADF_series, summed_ADF, _ = align_series_ML(ADF_images_list)

    return (aligned_BF_series, summed_BF), (aligned_ADF_series, summed_ADF)




def align_series_ML(image_series): #single series in a list

    initial_image = image_series[0]
    initial_image_shape = initial_image.shape
    shifts = []
    translated_list = []
    for image in tqdm(range(len(image_series))):
        translated_image = image_series[image].astype(np.float64)
        registration_values = registration_model(np.concatenate([initial_image,translated_image],axis=1), 'TEMRegistration', host=host, port='7443', image_encoder='.tiff')
        translation_values = registration_values[0]["translation"] #normalised between 0,1
        x_pixels_shift = translation_values[0]*initial_image_shape[0]
        y_pixels_shift = translation_values[1]*initial_image_shape[1]
        shifts.append((x_pixels_shift,y_pixels_shift))

        matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]])
        transposed_image = cv2.warpAffine(translated_image.astype(np.uint16),matrix,(initial_image_shape[1],initial_image_shape[0]))
        translated_list.append(transposed_image)

    summing_array = np.asarray(translated_list)
    summed_image = np.sum(summing_array, 0, dtype=np.float64)
    return translated_list,summed_image,shifts

def align_series_cross_correlation(image_series):
    initial_image = image_series[0]
    initial_image_shape = initial_image.shape
    fov_px = initial_image_shape[0]
    offsets = []
    correlation_coefficients = []
    translated_list = []
    for image in tqdm(range(len(image_series))):
        offset,coeffs = get_offset_of_pictures(initial_image,image_series[image],fov=fov_px,get_corr_coeff=True)
        offsets.append(offset)
        correlation_coefficients.append(coeffs)
        x_pixels_shift = offset[0]
        y_pixels_shift = offset[1]
        shift_matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]])
        translated_image = cv2.warpAffine(image_series[image].astype(np.uint16),shift_matrix,(initial_image_shape[1],initial_image_shape[0]))
        translated_list.append(translated_image)

    return translated_list,offsets,correlation_coefficients