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

from expert_pi.RSTEM.utilities import get_microscope_parameters
#registration_model(np.concatenate([original_image,translated_image],axis=1, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#registration_model(image, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#super_resolution_model(image=image, model_name='SwinIRImageDenoiser', host='172.19.1.16', port='7447') #for denoising
#manager = TorchserveRestManager(inference_port='8600', management_port='8081', host='172.19.1.16', image_encoder='.png')
#manager.infer(image=image, model_name='spot_segmentation') #spot detection
#manager.list_models()

#Tested works fine
#TODO sensible decimal places
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
        #print(f"Setting defocus")
        grpc_client.illumination.set_condenser_defocus(i,CFT.C3) #get correct command
        defocus_now = grpc_client.illumination.get_condenser_defocus(CFT.C3)

        defocus_offset = current_defocus-defocus_now
        defocus_offsets.append(defocus_offset)
        #print(f"Defocus offset {defocus_offset*1e9} nm")

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

#TODO tested ok
def acquire_STEM(BF=True,ADF=False, fov_um=None,pixel_time_us=None,num_pixels=1024,scan_rotation_deg=None):
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
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6
    if fov_um is not None:
        grpc_client.scanning.set_field_width(fov_um/1e6) #in microns

    if BF == True and ADF == True:
        detectors = [DT.BF,DT.HAADF]
        if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(
                DT.HAADF) is False:
            grpc_client.stem_detector.set_is_inserted(DT.BF,True) #insert BF detector
            grpc_client.stem_detector.set_is_inserted(DT.HAADF, True) #insert ADF detector
            print("Inserting requested detectors")
            print("Stabilising")
            sleep(5)
    elif BF == True and ADF == False:
        detectors = [DT.BF]
        if grpc_client.stem_detector.get_is_inserted(DT.BF) is False:
            grpc_client.projection.set_is_off_axis_stem_enabled(True)
            print("BF detector was not inserted, using off axis conditions")
    elif ADF == True and BF == False:
        detectors = [DT.HAADF]
        if grpc_client.stem_detector.get_is_inserted(DT.HAADF) is False:
            grpc_client.stem_detector.set_is_inserted(DT.HAADF, True)  # insert ADF detector
            print("Inserting requested detectors")
            print("Stabilising")
            sleep(5)

    scan_rotation_deg = np.rad2deg(grpc_client.scanning.get_rotation())

    metadata = get_microscope_parameters(scan_width_px = num_pixels,use_precession=False,camera_frequency_hz = None,
                                         STEM_dwell_time = pixel_time_us,scan_rotation=scan_rotation_deg)

    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=detectors)
    header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive small measurements in one chunk

    if BF == True and ADF == True:
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
        return(BF_image,ADF_image,metadata)

    if BF == True and ADF == False:
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
        return(BF_image,metadata)

    if ADF == True and BF == False:
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
        return(ADF_image,metadata)

#TODO tested ok
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



#TODO test again
def ML_drift_corrected_imaging(num_frames, pixel_time_us=None,series_output=False,num_pixels=None):
    #scan_rotation=0 #TODO is this scan rotation part really needed?
    #R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
    #              [-np.sin(scan_rotation), np.cos(scan_rotation)]])
    #grpc_client.scanning.set_rotation(scan_rotation)

    if pixel_time_us==None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    print("Pixel time",int(pixel_time*1e9),"ns")
    fov = grpc_client.scanning.get_field_width() #in meters
    print("Field of View in m",fov)

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
        print(s["x"],"X deflectors")
        print(s["y"],"Y deflectors")
        # apply drift correction between images:
        registration = registration_model(np.concatenate([BF_images_list[-1],BF_image],axis=1), 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #measure offset of images
        raw_shift = registration[0]["translation"]
        real_shift_x = raw_shift[0]*fov #shifts normalised between 0,1 proportion of image, convert to meters
        real_shift_y = raw_shift[1]*fov
        print(raw_shift)
        print("X shift m",real_shift_x)
        print("Y shift m",real_shift_y)
        #shift = np.dot(R, shift)  # rotate shifts back to scanning axes
        #print(i, (s0['x'] - s['x']) * 1e9, (s0['y'] - s['y']) * 1e9)
        grpc_client.illumination.set_shift(
                {"x": s['x'] + real_shift_x , "y": s['y'] + real_shift_y },grpc_client.illumination.DeflectorType.Scan) #apply shifts in microns to existing shifts


    fine_aligned_BF,summed_BF,_ = align_series(BF_images_list)
    fine_aligned_ADF,summed_ADF,_ = align_series(ADF_images_list)





    print("Image series output")
    return fine_aligned_BF,summed_BF, fine_aligned_ADF, summed_ADF

def acquire_series(num_frames, pixel_time_us=None, series_output=False,num_pixels=None ):
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




    if series_output == False:
        print("Summed image output")
        BF_images_array = np.asarray(BF_images_list)
        ADF_images_array = np.asarray(ADF_images_list)
        BF_summed = np.sum(BF_images_array, axis=0, dtype=np.float64)
        ADF_summed = np.sum(ADF_images_array, axis=0, dtype=np.float64)
        return BF_summed, ADF_summed

    if series_output == True:
        print("Image series output")
        return BF_images_list, ADF_images_list

def align_series(image_series): #single series in a list

    initial_image = image_series[0]
    initial_image_shape = initial_image.shape
    shifts = []
    translated_list = []
    for image in tqdm(range(len(image_series))):
        translated_image = image_series[image].astype(np.float64)
        registration_values = registration_model(np.concatenate([initial_image,translated_image],axis=1), 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.tiff')
        translation_values = registration_values[0]["translation"] #normalised between 0,1
        x_pixels_shift = translation_values[0]*initial_image_shape[0]
        y_pixels_shift = translation_values[1]*initial_image_shape[1]
        shifts.append((x_pixels_shift,y_pixels_shift))

        matrix = np.float32([[1,0,x_pixels_shift],[0,1,y_pixels_shift]])
        transposed_image = cv2.warpAffine(translated_image.astype(np.uint16),matrix,(initial_image_shape[1],initial_image_shape[0]))
        translated_list.append(transposed_image)

    summing_array = np.asarray(translated_list)
    summed_image = np.sum(summing_array, 0, dtype=np.float64)

    #summed_image.astype(np.float64)
    return translated_list,summed_image,shifts
