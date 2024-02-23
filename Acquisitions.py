from expert_pi.__main__ import window
from expert_pi import grpc_client
from expert_pi.controllers import scan_helper
from expert_pi.stream_clients import cache_client
from expert_pi.grpc_client.modules._common import DetectorType as DT
from stem_measurements import shift_measurements
import numpy as np
from time import sleep
import easygui as g
import cv2 as cv2
import fnmatch
import os

from serving_manager.api import registration_model
from serving_manager.api import super_resolution_model
from serving_manager.api import TorchserveRestManager

#from utilities import get_microscope_parameters

#registration_model(image, 'TEMRegistration', host='172.16.2.86', port='7443', image_encoder='.png') #TIFF is also ok
#super_resolution_model(image=image, model_name='SwinIRImageDenoiser', host='172.19.1.16', port='7447') #for denoising
#manager = TorchserveRestManager(inference_port='8600', management_port='8081', host='172.19.1.16', image_encoder='.png')
#manager.infer(image=image, model_name='spot_segmentation') #spot detection
#manager.list_models()


def acquire_focal_series(extent_nm,steps,BF=True,ADF=True,num_pixels=1024,pixel_time_us=5):
    current_defocus = grpc_client.illumination.get_condenser_defocus()
    lower_defocus = current_defocus-(extent_nm*1e-9/2)
    higher_defocus = current_defocus+(extent_nm*1e-9/2)
    defocus_intervals = np.linspace(lower_defocus,higher_defocus,steps)

    image_series = []
    for i in defocus_intervals:
        print(f"Setting defocus to {i}")
        grpc_client.illumination.set_condenser_defocus(i) #get correct command
        defocus_now = grpc_client.illumination.get_defocus()

        defocus_offset = defocus_now-i

        print(f"Defocus now {defocus_now*1e9} nm")
        print(f"Defocus offset {defocus_offset*1e9} nm")
        scan_id = scan_helper.start_rectangle_scan(pixel_time=(pixel_time_us*1e-6), total_size=num_pixels, frames=1,
                                                   detectors=(DT.BF,DT.HAADF))
        header, data = cache_client.get_item(scan_id, num_pixels**2)  # retrive small measurements in one chunk

        if BF==True and ADF_image==False:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            image_series.append(BF_image)
        if ADF==True and BF == False:
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append(ADF_image)
        if BF == True and ADF == True:
            BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
            ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
            image_series.append((BF_image,ADF_image))

    return image_series,defocus_intervals

#TODO untested
def acquire_STEM(signals = ["BF","ADF"], fov=None,pixel_time_us=None,num_pixels=1024,scan_rotation=None):
    """Acquires a single STEM image from the requested detectors
    returns a tuple with the images and the metadata in a dictionary
    Parameters
    signals: list of signals in string form ["BF","ADF"]
    fov : field of view in microns
    pixel_time_us: dwell time in microseconds
    num_pixels: scan width in pixels,
    scan_rotation: scan rotation in degrees"""

    if scan_rotation is not None:
        grpc_client.scanning.set_rotation(np.deg2rad(scan_rotation))
    if pixel_time_us is None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6
    if fov is not None:
        grpc_client.scanning.set_field_width(fov) #in microns

    if "BF" and "ADF" in signals:
        detectors = [DT.BF,DT.HAADF]
        if grpc_client.stem_detector.get_is_inserted(DT.BF) or grpc_client.stem_detector.get_is_inserted(
                DT.HAADF) is False:
            grpc_client.stem_detector.set_is_inserted(DT.BF,True) #insert BF detector
            grpc_client.stem_detector.set_is_inserted(DT.HAADF, True) #insert ADF detector
            print("Inserting requested detectors")
            sleep(5)
    elif "BF" and not "ADF" in signals:
        detectors = [DT.BF]
        if grpc_client.stem_detector.get_is_inserted(DT.BF) is False:
            grpc_client.stem_detector.set_is_inserted(DT.BF, True)  # insert BF detector
            print("Inserting requested detectors")
            sleep(5)
    elif "ADF" and not "BF" in signals:
        detectors = [DT.HAADF]
        if grpc_client.stem_detector.get_is_inserted(DT.HAADF) is False:
            grpc_client.stem_detector.set_is_inserted(DT.HAADF, True)  # insert ADF detector
            print("Inserting requested detectors")
            sleep(5)

    metadata = get_microscope_parameters(num_pixels,False,False,
                                         pixel_time_us,scan_rotation=scan_rotation)

    scan_id = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=detectors)
    header, data = cache_client.get_item(scan_id, num_pixels ** 2) #retrive small measurements in one chunk

    if "BF" and "ADF" in signals:
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
        return(BF_image,ADF_image,metadata)

    if "BF" and not "ADF" in signals:
        BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
        return(BF_image,metadata)

    if "ADF" and not "BF" in signals:
        ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
        return(ADF_image,metadata)

#TODO untested
def save_STEM(image,name=None,folder=None):
    if folder == None:
        folder = g.diropenbox("Enter save location","Enter save location")
        folder + "\\"
    if name == None:
        num_files_in_dir = len(fnmatch.filter(os.listdir(folder), '*.tiff'))
        name = f"STEM000{num_files_in_dir+1}" #should increment the image number
    name = name+".tiff"
    filename = str(folder+name)
    print(f"Saving {name} to {folder}")
    cv2.imwrite(filename,image)


#TODO test again
def drift_corrected_imaging(num_frames, pixel_time_us=None,series_output=False,shift_method="patches",num_pixels=None):
    scan_rotation=0 #TODO is this scan rotation part really needed?
    R = np.array([[np.cos(scan_rotation), np.sin(scan_rotation)],
                  [-np.sin(scan_rotation), np.cos(scan_rotation)]])
    grpc_client.scanning.set_rotation(scan_rotation)

    if pixel_time_us==None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    print("Pixel time",int(pixel_time*1e9),"ns")
    fov = grpc_client.scanning.get_field_width() #in microns
    print("Field of View in m",fov)

    #TODO add in option to calculate series dose?

    if num_pixels == None:
        num_pixels = 1024

    BF_images_list = []
    ADF_images_list = []
    image_offsets = []

    initial_scan = scan_helper.start_rectangle_scan(pixel_time=pixel_time, total_size=num_pixels, frames=1,
                                               detectors=[DT.BF, DT.HAADF])
    header, data = cache_client.get_item(initial_scan, num_pixels ** 2) #retrive small measurements in one chunk

    initial_shift = grpc_client.illumination.get_shift(grpc_client.illumination.DeflectorType.Scan)

    initial_BF_image = data["stemData"]["BF"].reshape(num_pixels, num_pixels)
    BF_images_list.append(initial_BF_image)
    initial_ADF_image = data["stemData"]["HAADF"].reshape(num_pixels, num_pixels)
    ADF_images_list.append(initial_ADF_image)
    image_offsets.append(initial_shift)

    #TODO check other shift measurement methods and add them properly
    if shift_method == "patches":
        shift_method = shift_measurements.Method.PatchesPass2
    elif shift_method =="ML":
        shift_method = shift_measurements.Method.registration_model

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
        shift = shift_measurements.get_offset_of_pictures(initial_BF_image, BF_image, fov, method=shift_method) #measure offset of images
        shift = np.dot(R, shift)  # rotate shifts back to scanning axes
        #print(i, (s0['x'] - s['x']) * 1e9, (s0['y'] - s['y']) * 1e9)
        grpc_client.illumination.set_shift(
                {"x": s['x'] - shift[0] * 1e-6, "y": s['y'] - shift[1] * 1e-6},grpc_client.illumination.DeflectorType.Scan) #apply shifts in microns to existing shifts

    BF_images_array = np.asarray(BF_images_list)
    ADF_images_array = np.asarray(ADF_images_list)

    BF_summed = np.sum(BF_images_array,axis=0,dtype=np.float64)
    ADF_summed = np.sum(ADF_images_array,axis=0,dtype=np.float64)

    if series_output ==  False:
        print("Summed image output")
        return BF_summed, ADF_summed


    if series_output == True:
        print("Image series output")
        return BF_images_list, ADF_images_list

def acquire_series(num_frames, pixel_time_us=None, series_output=False,num_pixels=None ):

    if pixel_time_us == None:
        pixel_time = window.scanning.pixel_time_spin.value()/1e6  # gets current pixel time from UI and convert to seconds
    else:
        pixel_time = pixel_time_us/1e6

    print("Pixel time", int(pixel_time*1e9), "ns")
    fov = grpc_client.scanning.get_field_width()  # in microns
    print("Field of View in um", fov)

    # TODO add in option to calculate series dose?

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
